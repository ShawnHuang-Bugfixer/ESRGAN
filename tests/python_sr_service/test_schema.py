import pytest

from python_sr_service.domain.errors import ErrorCode, ServiceError
from python_sr_service.domain.schema import TaskMessage


def _payload(timestamp: str):
    return {
        'schemaVersion': '1.0',
        'eventId': 'evt_task_1',
        'timestamp': timestamp,
        'taskId': 1,
        'taskNo': 'SR202602241401057558586',
        'userId': 2,
        'type': 'image',
        'inputFileKey': '/space/in.webp',
        'scale': 4,
        'modelName': 'RealESRGAN_x4plus',
        'modelVersion': 'v1.0.0',
        'attempt': 1,
        'traceId': 'trace_1',
    }


def test_task_message_accepts_nanosecond_timestamp():
    task = TaskMessage.from_dict(_payload('2026-02-24T06:01:05.759187600Z'))
    assert task.timestamp == '2026-02-24T06:01:05.759187600Z'


def test_task_message_rejects_invalid_timestamp():
    with pytest.raises(ServiceError) as exc_info:
        TaskMessage.from_dict(_payload('2026-02-24 25:01:05'))

    assert exc_info.value.code == ErrorCode.SCHEMA_INVALID
    assert str(exc_info.value) == 'Invalid timestamp format, expected ISO-8601'
