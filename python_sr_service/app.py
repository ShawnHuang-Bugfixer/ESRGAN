import logging
from pathlib import Path
import shutil

import torch

from python_sr_service.config import Settings
from python_sr_service.domain.errors import ServiceError
from python_sr_service.pipeline.video_pipeline import available_video_encoders, resolve_video_codec_chain
from python_sr_service.idempotency.redis_store import RedisIdempotencyStore
from python_sr_service.runtime.logging import format_log_fields, setup_logging
from python_sr_service.worker.consumer import RabbitMQConsumer
from python_sr_service.worker.publisher import RabbitMQResultPublisher


def main() -> None:
    settings = Settings.from_env()
    setup_logging(settings.runtime.log_level)
    logger = logging.getLogger(__name__)
    logger.info('python_sr_service_start')
    _validate_video_runtime(logger, settings)
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




def _validate_video_runtime(logger: logging.Logger, settings: Settings) -> None:
    if not settings.inference.video_enabled:
        return

    missing: list[dict[str, str]] = []
    for field_name in ('ffmpeg_bin', 'ffprobe_bin'):
        command = str(getattr(settings.inference, field_name, '') or '').strip()
        if not command:
            missing.append({'field': field_name, 'configured': command})
            continue

        resolved = shutil.which(command)
        if resolved or Path(command).is_file():
            continue

        missing.append({'field': field_name, 'configured': command})

    if missing:
        raise RuntimeError(f'Video runtime binaries missing: {missing}')

    available_encoders = available_video_encoders(settings.inference.ffmpeg_bin)
    if not available_encoders:
        raise RuntimeError('Failed to enumerate video encoders from ffmpeg')

    try:
        resolved_codec_chain = resolve_video_codec_chain(settings.inference)
    except ServiceError as exc:
        raise RuntimeError(str(exc)) from exc

    logger.info(
        'video_runtime_ready %s',
        format_log_fields(
            {
                'ffmpegBin': settings.inference.ffmpeg_bin,
                'ffprobeBin': settings.inference.ffprobe_bin,
                'videoEnabled': settings.inference.video_enabled,
                'configuredVideoCodec': settings.inference.video_codec,
                'availableVideoEncoders': available_encoders,
                'resolvedVideoCodecChain': resolved_codec_chain,
            },
        ),
    )

def _log_gpu_runtime(logger: logging.Logger, settings: Settings) -> None:
    device_text = settings.inference.device.strip()
    lower = device_text.lower()
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    fields = {
        'phase': 'startup',
        'configuredDevice': device_text,
        'imageModel': settings.inference.model_name,
        'videoModel': settings.inference.video_model_name,
        'imageTile': settings.inference.tile,
        'videoTile': settings.inference.video_tile,
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
