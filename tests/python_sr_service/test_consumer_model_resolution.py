import types

from python_sr_service.config import InferenceSettings
from python_sr_service.domain.schema import TaskMessage
from python_sr_service.worker.consumer import RabbitMQConsumer


def _payload(task_type: str, model_name: str):
    return {
        'schemaVersion': '1.0',
        'eventId': f'evt_{task_type}_1',
        'timestamp': '2026-03-22T08:00:00Z',
        'taskId': 1,
        'taskNo': 'SR202603220800000001',
        'userId': 2,
        'type': task_type,
        'inputFileKey': 'space/demo.mp4' if task_type == 'video' else 'space/demo.png',
        'scale': 2,
        'modelName': model_name,
        'modelVersion': 'v1.0.0',
        'attempt': 1,
        'traceId': 'trace_1',
    }


def test_resolve_model_name_uses_active_video_fallback_model():
    consumer = RabbitMQConsumer.__new__(RabbitMQConsumer)
    consumer._settings = types.SimpleNamespace(
        inference=InferenceSettings(
            model_name='RealESRGAN_x4plus',
            video_model_name='realesr-general-x4v3',
        ),
    )
    consumer._video_pipeline = types.SimpleNamespace(active_model_name='RealESRGAN_x4plus')
    consumer._worker_id = 'worker-test'
    consumer._task_log = lambda task, **extra: ''

    task = TaskMessage.from_dict(_payload('video', 'realesr-general-x4v3'))

    assert consumer._resolve_model_name(task) == 'RealESRGAN_x4plus'


def test_resolve_model_name_keeps_image_model_resolution():
    consumer = RabbitMQConsumer.__new__(RabbitMQConsumer)
    consumer._settings = types.SimpleNamespace(
        inference=InferenceSettings(
            model_name='RealESRGAN_x4plus',
            video_model_name='realesr-general-x4v3',
        ),
    )
    consumer._video_pipeline = types.SimpleNamespace(active_model_name='RealESRGAN_x4plus')
    consumer._worker_id = 'worker-test'
    consumer._task_log = lambda task, **extra: ''

    task = TaskMessage.from_dict(_payload('image', 'RealESRGAN_x4plus'))

    assert consumer._resolve_model_name(task) == 'RealESRGAN_x4plus'
