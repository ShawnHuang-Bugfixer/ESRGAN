import pytest

from python_sr_service.config import COSSettings
from python_sr_service.domain.errors import ErrorCode, ServiceError
from python_sr_service.storage.cos_client import TencentCOSClient


class FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def get_stream_to_file(self, path: str):
        with open(path, 'wb') as file_obj:
            file_obj.write(self._data)


class FakeCosError(Exception):
    def __init__(self, status_code: int, message: str = 'cos error'):
        super().__init__(message)
        self.status_code = status_code


class FakeCosClient:
    def __init__(self):
        self.storage = {}
        self.raise_get = None
        self.raise_put = None
        self.raise_head = None
        self.upload_file_calls = []

    def get_object(self, Bucket, Key):
        if self.raise_get:
            raise self.raise_get
        if Key not in self.storage:
            raise FakeCosError(404, 'not found')
        return {'Body': FakeBody(self.storage[Key])}

    def put_object(self, Bucket, Body, Key):
        if self.raise_put:
            raise self.raise_put
        self.storage[Key] = Body.read()
        return {'ETag': 'etag'}

    def upload_file(self, Bucket, Key, LocalFilePath, PartSize=1, MAXThread=5, **kwargs):
        if self.raise_put:
            raise self.raise_put
        with open(LocalFilePath, 'rb') as file_obj:
            self.storage[Key] = file_obj.read()
        self.upload_file_calls.append(
            {
                'Bucket': Bucket,
                'Key': Key,
                'LocalFilePath': LocalFilePath,
                'PartSize': PartSize,
                'MAXThread': MAXThread,
            },
        )
        return {'ETag': 'etag-multipart'}

    def head_object(self, Bucket, Key):
        if self.raise_head:
            raise self.raise_head
        if Key not in self.storage:
            raise FakeCosError(404, 'not found')
        return {}

    def delete_object(self, Bucket, Key):
        self.storage.pop(Key, None)

    def list_objects_v2(self, Bucket, Prefix):
        keys = [key for key in self.storage if key.startswith(Prefix)]
        return {'Contents': [{'Key': key} for key in keys]}


class FakeCosClientLegacyList(FakeCosClient):
    def list_objects_v2(self, Bucket, Prefix):
        raise AttributeError('list_objects_v2 is not available')

    def list_objects(self, Bucket, Prefix):
        keys = [key for key in self.storage if key.startswith(Prefix)]
        return {'Contents': [{'Key': key} for key in keys]}


def _settings(prefix=''):
    return COSSettings(
        secret_id='id',
        secret_key='key',
        region='ap-guangzhou',
        bucket='bucket-1250000000',
        prefix=prefix,
        timeout_seconds=120,
        multipart_threshold_mb=8,
        upload_part_mb=4,
        upload_max_thread=2,
    )


def test_upload_download_roundtrip(tmp_path):
    cos_client = FakeCosClient()
    client = TencentCOSClient(_settings(prefix='sr'), client=cos_client)

    source_file = tmp_path / 'input.bin'
    source_file.write_bytes(b'hello cos')
    client.upload(str(source_file), 'task/input.bin')

    output_file = tmp_path / 'output.bin'
    client.download('task/input.bin', str(output_file))
    assert output_file.read_bytes() == b'hello cos'


def test_upload_large_file_uses_multipart(tmp_path):
    cos_client = FakeCosClient()
    settings = COSSettings(
        secret_id='id',
        secret_key='key',
        region='ap-guangzhou',
        bucket='bucket-1250000000',
        multipart_threshold_mb=1,
        upload_part_mb=2,
        upload_max_thread=3,
    )
    client = TencentCOSClient(settings, client=cos_client)

    source_file = tmp_path / 'big.bin'
    source_file.write_bytes(b'x' * (2 * 1024 * 1024))
    client.upload(str(source_file), 'task/big.bin')

    assert len(cos_client.upload_file_calls) == 1
    call = cos_client.upload_file_calls[0]
    assert call['PartSize'] == 2
    assert call['MAXThread'] == 3


def test_download_missing_maps_to_input_not_found(tmp_path):
    client = TencentCOSClient(_settings(), client=FakeCosClient())

    with pytest.raises(ServiceError) as exc_info:
        client.download('missing/file.png', str(tmp_path / 'file.png'))

    assert exc_info.value.code == ErrorCode.INPUT_NOT_FOUND


def test_upload_failure_maps_to_cos_upload_failed(tmp_path):
    cos_client = FakeCosClient()
    cos_client.raise_put = FakeCosError(500, 'upload failed')
    client = TencentCOSClient(_settings(), client=cos_client)

    source_file = tmp_path / 'input.bin'
    source_file.write_bytes(b'x')

    with pytest.raises(ServiceError) as exc_info:
        client.upload(str(source_file), 'path/file.bin')

    assert exc_info.value.code == ErrorCode.COS_UPLOAD_FAILED
    assert exc_info.value.retryable is True


def test_exists_delete_and_list(tmp_path):
    cos_client = FakeCosClient()
    client = TencentCOSClient(_settings(prefix='sr'), client=cos_client)

    file_path = tmp_path / 'image.bin'
    file_path.write_bytes(b'data')
    client.upload(str(file_path), 'a/image.bin')

    assert client.exists('a/image.bin') is True
    assert client.exists('a/missing.bin') is False

    keys = client.list_objects('a')
    assert keys == ['sr/a/image.bin']

    client.delete('a/image.bin')
    assert client.exists('a/image.bin') is False


def test_list_objects_fallback_to_legacy_api(tmp_path):
    cos_client = FakeCosClientLegacyList()
    client = TencentCOSClient(_settings(prefix='sr'), client=cos_client)

    file_path = tmp_path / 'image.bin'
    file_path.write_bytes(b'data')
    client.upload(str(file_path), 'legacy/image.bin')

    keys = client.list_objects('legacy')
    assert keys == ['sr/legacy/image.bin']
