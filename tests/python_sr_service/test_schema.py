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


def test_task_message_video_options_default_values():
    payload = _payload('2026-02-24T06:01:05.759187600Z')
    payload['type'] = 'video'

    task = TaskMessage.from_dict(payload)

    assert task.task_type == 'video'
    assert task.video_options.keep_audio is True
    assert task.video_options.extract_frame_first is True
    assert task.video_options.fps_override is None


def test_task_message_video_options_with_values():
    payload = _payload('2026-02-24T06:01:05.759187600Z')
    payload['type'] = 'video'
    payload['videoOptions'] = {
        'keepAudio': False,
        'extractFrameFirst': True,
        'fpsOverride': 25,
    }

    task = TaskMessage.from_dict(payload)

    assert task.video_options.keep_audio is False
    assert task.video_options.extract_frame_first is True
    assert task.video_options.fps_override == 25.0


def test_task_message_video_options_rejects_invalid_value():
    payload = _payload('2026-02-24T06:01:05.759187600Z')
    payload['videoOptions'] = {'keepAudio': 'yes'}

    with pytest.raises(ServiceError) as exc_info:
        TaskMessage.from_dict(payload)

    assert exc_info.value.code == ErrorCode.SCHEMA_INVALID
    assert 'keepAudio' in str(exc_info.value)
