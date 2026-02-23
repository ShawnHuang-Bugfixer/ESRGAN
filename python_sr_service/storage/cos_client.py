import os
from typing import Any, Optional

from python_sr_service.config import COSSettings
from python_sr_service.domain.errors import ErrorCode, ServiceError
from python_sr_service.storage.object_storage import ObjectStorage


class TencentCOSClient(ObjectStorage):
    def __init__(
        self,
        settings: COSSettings,
        client: Optional[Any] = None,
    ):
        self._settings = settings
        self._bucket = settings.bucket
        self._prefix = settings.prefix.strip('/')
        self._client = client or self._create_client(settings)

    def download(self, object_key: str, local_path: str) -> None:
        full_key = self._full_key(object_key)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=full_key)
            os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
            response['Body'].get_stream_to_file(local_path)
        except Exception as exc:
            raise self._to_service_error('download', full_key, exc) from exc

    def upload(self, local_path: str, object_key: str) -> None:
        full_key = self._full_key(object_key)
        try:
            with open(local_path, 'rb') as file_obj:
                self._client.put_object(Bucket=self._bucket, Body=file_obj, Key=full_key)
        except Exception as exc:
            raise self._to_service_error('upload', full_key, exc) from exc

    def exists(self, object_key: str) -> bool:
        full_key = self._full_key(object_key)
        try:
            self._client.head_object(Bucket=self._bucket, Key=full_key)
            return True
        except Exception as exc:
            status_code = _get_status_code(exc)
            if status_code == 404:
                return False
            raise self._to_service_error('download', full_key, exc) from exc

    def delete(self, object_key: str) -> None:
        full_key = self._full_key(object_key)
        try:
            self._client.delete_object(Bucket=self._bucket, Key=full_key)
        except Exception as exc:
            raise self._to_service_error('upload', full_key, exc) from exc

    def list_objects(self, prefix: str):
        full_prefix = self._full_key(prefix)
        try:
            response = None
            if hasattr(self._client, 'list_objects_v2'):
                try:
                    response = self._client.list_objects_v2(Bucket=self._bucket, Prefix=full_prefix)
                except AttributeError:
                    response = None
            if response is None:
                response = self._client.list_objects(Bucket=self._bucket, Prefix=full_prefix)
            contents = response.get('Contents', [])
            return [item['Key'] for item in contents]
        except Exception as exc:
            raise self._to_service_error('download', full_prefix, exc) from exc

    def _full_key(self, object_key: str) -> str:
        normalized = object_key.lstrip('/')
        if not self._prefix:
            return normalized
        return f'{self._prefix}/{normalized}' if normalized else self._prefix

    def _create_client(self, settings: COSSettings):
        try:
            from qcloud_cos import CosConfig, CosS3Client
        except ImportError as exc:
            raise RuntimeError('cos-python-sdk-v5 is required for Tencent COS integration') from exc

        config_kwargs = {
            'Region': settings.region,
            'SecretId': settings.secret_id,
            'SecretKey': settings.secret_key,
            'Scheme': settings.scheme,
        }
        if settings.token:
            config_kwargs['Token'] = settings.token
        if settings.endpoint:
            config_kwargs['Endpoint'] = settings.endpoint

        config = CosConfig(**config_kwargs)
        return CosS3Client(config)

    def _to_service_error(self, operation: str, object_key: str, exc: Exception) -> ServiceError:
        status_code = _get_status_code(exc)
        retryable = status_code in (429, 500, 502, 503, 504)
        if operation == 'download' and status_code == 404:
            code = ErrorCode.INPUT_NOT_FOUND
            message = f'Input file not found in COS: {object_key}'
        elif operation == 'upload':
            code = ErrorCode.COS_UPLOAD_FAILED
            message = f'Failed to upload file to COS: {object_key}'
        else:
            code = ErrorCode.COS_DOWNLOAD_FAILED
            message = f'Failed to download file from COS: {object_key}'
        return ServiceError(code=code, message=message, retryable=retryable, cause=exc)


def _get_status_code(exc: Exception):
    value = getattr(exc, 'status_code', None)
    if value:
        return value
    getter = getattr(exc, 'get_status_code', None)
    if callable(getter):
        return getter()
    return None
