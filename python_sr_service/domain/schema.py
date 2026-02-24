from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, Dict

from python_sr_service.domain.errors import ErrorCode, ServiceError


@dataclass(frozen=True)
class TaskMessage:
    schema_version: str
    event_id: str
    timestamp: str
    task_id: int
    task_no: str
    user_id: int
    task_type: str
    input_file_key: str
    scale: int
    model_name: str
    model_version: str
    attempt: int
    trace_id: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'TaskMessage':
        required_fields = (
            'schemaVersion',
            'eventId',
            'timestamp',
            'taskId',
            'taskNo',
            'userId',
            'type',
            'inputFileKey',
            'scale',
            'modelName',
            'modelVersion',
            'attempt',
            'traceId',
        )
        missing = [field for field in required_fields if field not in payload]
        if missing:
            raise ServiceError(
                code=ErrorCode.SCHEMA_INVALID,
                message=f'Missing required fields: {", ".join(missing)}',
                retryable=False,
            )

        _validate_timestamp(payload['timestamp'])

        try:
            return cls(
                schema_version=str(payload['schemaVersion']),
                event_id=str(payload['eventId']),
                timestamp=str(payload['timestamp']),
                task_id=int(payload['taskId']),
                task_no=str(payload['taskNo']),
                user_id=int(payload['userId']),
                task_type=str(payload['type']),
                input_file_key=str(payload['inputFileKey']),
                scale=int(payload['scale']),
                model_name=str(payload['modelName']),
                model_version=str(payload['modelVersion']),
                attempt=int(payload['attempt']),
                trace_id=str(payload['traceId']),
            )
        except (TypeError, ValueError) as exc:
            raise ServiceError(
                code=ErrorCode.SCHEMA_INVALID,
                message='Invalid value type in task payload',
                retryable=False,
                cause=exc,
            ) from exc


def _validate_timestamp(timestamp_text: str) -> None:
    normalized = _normalize_timestamp(timestamp_text)
    try:
        datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ServiceError(
            code=ErrorCode.SCHEMA_INVALID,
            message='Invalid timestamp format, expected ISO-8601',
            retryable=False,
            cause=exc,
        ) from exc


def _normalize_timestamp(timestamp_text: str) -> str:
    normalized = str(timestamp_text).replace('Z', '+00:00')
    # Python 3.10 only supports microseconds (6 digits), truncate higher precision.
    return re.sub(r'(\.\d{6})\d+(?=(?:[+-]\d{2}:\d{2})?$)', r'\1', normalized)
