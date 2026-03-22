from dataclasses import replace
import logging

import torch

from python_sr_service.config import Settings
from python_sr_service.idempotency.redis_store import RedisIdempotencyStore
from python_sr_service.runtime.logging import format_log_fields, setup_logging
from python_sr_service.worker.consumer import LOCKED_SERVICE_MODEL_NAME, RabbitMQConsumer
from python_sr_service.worker.publisher import RabbitMQResultPublisher


def main() -> None:
    settings = Settings.from_env()
    settings = replace(
        settings,
        inference=replace(settings.inference, model_name=LOCKED_SERVICE_MODEL_NAME),
    )
    setup_logging(settings.runtime.log_level)
    logger = logging.getLogger(__name__)
    logger.info('python_sr_service_start')
    _log_gpu_runtime(logger, settings)
    idempotency_store = RedisIdempotencyStore(
        redis_url=settings.redis.url,
        ttl_seconds=settings.idempotency.ttl_seconds,
    )
    publisher = RabbitMQResultPublisher(settings.rabbitmq)
    consumer = RabbitMQConsumer(
        settings=settings,
        idempotency_store=idempotency_store,
        publisher=publisher,
    )
    consumer.prepare()
    consumer.start()
    logger.info('python_sr_service_stop')


def _log_gpu_runtime(logger: logging.Logger, settings: Settings) -> None:
    device_text = settings.inference.device.strip()
    lower = device_text.lower()
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    fields = {
        'phase': 'startup',
        'configuredDevice': device_text,
        'effectiveModel': settings.inference.model_name,
        'tile': settings.inference.tile,
        'tilePad': settings.inference.tile_pad,
        'prePad': settings.inference.pre_pad,
        'fp32': settings.inference.fp32,
        'cudaAvailable': cuda_available,
        'gpuCount': gpu_count,
    }

    if lower == 'cpu' or not cuda_available or not lower.startswith('cuda'):
        logger.info('gpu_runtime %s', format_log_fields(fields))
        return

    gpu_id = 0
    if ':' in lower:
        try:
            gpu_id = int(lower.split(':', 1)[1])
        except ValueError:
            gpu_id = 0

    if gpu_id < 0 or gpu_id >= gpu_count:
        fields['resolvedGpuId'] = gpu_id
        fields['gpuError'] = 'configured gpu id is out of range'
        logger.warning('gpu_runtime %s', format_log_fields(fields))
        return

    props = torch.cuda.get_device_properties(gpu_id)
    fields.update(
        {
            'resolvedGpuId': gpu_id,
            'gpuName': torch.cuda.get_device_name(gpu_id),
            'totalMemoryMB': int(props.total_memory / (1024 * 1024)),
            'allocatedMemoryMB': int(torch.cuda.memory_allocated(gpu_id) / (1024 * 1024)),
            'reservedMemoryMB': int(torch.cuda.memory_reserved(gpu_id) / (1024 * 1024)),
            'multiProcessorCount': int(getattr(props, 'multi_processor_count', 0)),
            'cudaVersion': torch.version.cuda,
            'cudnnEnabled': bool(torch.backends.cudnn.enabled),
        },
    )
    logger.info('gpu_runtime %s', format_log_fields(fields))


if __name__ == '__main__':
    main()
