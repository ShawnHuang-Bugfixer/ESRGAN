from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent / 'application.yml')


@dataclass(frozen=True)
class COSSettings:
    secret_id: str
    secret_key: str
    region: str
    bucket: str
    scheme: str = 'https'
    token: str = ''
    prefix: str = ''
    endpoint: str = ''
    timeout_seconds: int = 120
    multipart_threshold_mb: int = 8
    upload_part_mb: int = 8
    upload_max_thread: int = 3


@dataclass(frozen=True)
class MySQLSettings:
    dsn: str = ''
    connect_timeout_seconds: int = 5
    read_timeout_seconds: int = 10
    write_timeout_seconds: int = 10
    autocommit: bool = True


@dataclass(frozen=True)
class RabbitMQSettings:
    url: str
    task_queue: str = 'sr.task.queue'
    task_exchange: str = 'x.sr.task.direct'
    task_routing_key: str = 'sr.task'
    result_exchange: str = 'x.sr.result.direct'
    result_routing_key: str = 'sr.result'
    retry_exchange: str = 'x.sr.retry.direct'
    prefetch: int = 1


@dataclass(frozen=True)
class RedisSettings:
    url: str


@dataclass(frozen=True)
class IdempotencySettings:
    ttl_seconds: int = 259200


@dataclass(frozen=True)
class InferenceSettings:
    model_name: str = 'RealESRGAN_x4plus'
    model_weights: str = ''
    denoise_strength: float = 1.0
    device: str = 'cuda:0'
    tile: int = 0
    tile_pad: int = 10
    pre_pad: int = 0
    fp32: bool = False
    video_enabled: bool = True
    ffmpeg_bin: str = 'ffmpeg'
    ffprobe_bin: str = 'ffprobe'
    max_video_frames: int = 3000
    max_video_seconds: int = 180
    video_processing_mode: str = 'stream'
    video_frame_ext: str = 'png'
    video_codec: str = 'h264_nvenc'
    video_codec_fallbacks: tuple[str, ...] = ('h264_mf', 'libx264', 'mpeg4')
    video_pix_fmt: str = 'yuv420p'
    audio_fallback_no_audio: bool = True


@dataclass(frozen=True)
class RuntimeSettings:
    work_dir: str = './runtime'
    worker_id: str = ''
    log_level: str = 'INFO'


@dataclass(frozen=True)
class Settings:
    cos: COSSettings
    mysql: MySQLSettings
    rabbitmq: RabbitMQSettings
    redis: RedisSettings
    idempotency: IdempotencySettings
    inference: InferenceSettings
    runtime: RuntimeSettings

    @classmethod
    def from_env(cls, config_path: str = ''):
        resolved_path = config_path or os.getenv('SR_CONFIG_FILE', DEFAULT_CONFIG_PATH)
        config_data = _load_config_file(resolved_path)
        cos_config = config_data.get('cos', {})
        mysql_config = config_data.get('mysql', {})
        rabbitmq_config = config_data.get('rabbitmq', {})
        redis_config = config_data.get('redis', {})
        idempotency_config = config_data.get('idempotency', {})
        inference_config = config_data.get('inference', {})
        runtime_config = config_data.get('runtime', {})

        return cls(
            cos=COSSettings(
                secret_id=_get_value('COS_SECRET_ID', cos_config, 'secret_id', required=True),
                secret_key=_get_value('COS_SECRET_KEY', cos_config, 'secret_key', required=True),
                region=_get_value('COS_REGION', cos_config, 'region', required=True),
                bucket=_get_value('COS_BUCKET', cos_config, 'bucket', required=True),
                scheme=_get_value('COS_SCHEME', cos_config, 'scheme', default='https'),
                token=_get_value('COS_TOKEN', cos_config, 'token', default=''),
                prefix=_get_value('COS_PREFIX', cos_config, 'prefix', default=''),
                endpoint=_get_value('COS_ENDPOINT', cos_config, 'endpoint', default=''),
                timeout_seconds=_get_int_value('COS_TIMEOUT_SECONDS', cos_config, 'timeout_seconds', default=120),
                multipart_threshold_mb=_get_int_value(
                    'COS_MULTIPART_THRESHOLD_MB',
                    cos_config,
                    'multipart_threshold_mb',
                    default=8,
                ),
                upload_part_mb=_get_int_value('COS_UPLOAD_PART_MB', cos_config, 'upload_part_mb', default=8),
                upload_max_thread=_get_int_value(
                    'COS_UPLOAD_MAX_THREAD',
                    cos_config,
                    'upload_max_thread',
                    default=3,
                ),
            ),
            mysql=MySQLSettings(
                dsn=_get_value('MYSQL_DSN', mysql_config, 'dsn', default=''),
                connect_timeout_seconds=_get_int_value(
                    'MYSQL_CONNECT_TIMEOUT_SECONDS',
                    mysql_config,
                    'connect_timeout_seconds',
                    default=5,
                ),
                read_timeout_seconds=_get_int_value(
                    'MYSQL_READ_TIMEOUT_SECONDS',
                    mysql_config,
                    'read_timeout_seconds',
                    default=10,
                ),
                write_timeout_seconds=_get_int_value(
                    'MYSQL_WRITE_TIMEOUT_SECONDS',
                    mysql_config,
                    'write_timeout_seconds',
                    default=10,
                ),
                autocommit=_get_bool_value(
                    'MYSQL_AUTOCOMMIT',
                    mysql_config,
                    'autocommit',
                    default=True,
                ),
            ),
            rabbitmq=RabbitMQSettings(
                url=_get_value('MQ_URL', rabbitmq_config, 'url', required=True),
                task_queue=_get_value('MQ_TASK_QUEUE', rabbitmq_config, 'task_queue', default='sr.task.queue'),
                task_exchange=_get_value('MQ_TASK_EXCHANGE', rabbitmq_config, 'task_exchange', default='x.sr.task.direct'),
                task_routing_key=_get_value('MQ_TASK_ROUTING_KEY', rabbitmq_config, 'task_routing_key', default='sr.task'),
                result_exchange=_get_value(
                    'MQ_RESULT_EXCHANGE',
                    rabbitmq_config,
                    'result_exchange',
                    default='x.sr.result.direct',
                ),
                result_routing_key=_get_value(
                    'MQ_RESULT_ROUTING_KEY',
                    rabbitmq_config,
                    'result_routing_key',
                    default='sr.result',
                ),
                retry_exchange=_get_value(
                    'MQ_RETRY_EXCHANGE',
                    rabbitmq_config,
                    'retry_exchange',
                    default='x.sr.retry.direct',
                ),
                prefetch=_get_int_value('MQ_PREFETCH', rabbitmq_config, 'prefetch', default=1),
            ),
            redis=RedisSettings(
                url=_get_value('REDIS_URL', redis_config, 'url', required=True),
            ),
            idempotency=IdempotencySettings(
                ttl_seconds=_get_int_value(
                    'IDEMP_TTL_SECONDS',
                    idempotency_config,
                    'ttl_seconds',
                    default=259200,
                ),
            ),
            inference=InferenceSettings(
                model_name=_get_value('MODEL_NAME', inference_config, 'model_name', default='RealESRGAN_x4plus'),
                model_weights=_get_value(
                    'MODEL_WEIGHTS',
                    inference_config,
                    'model_weights',
                    default='',
                ),
                denoise_strength=_get_float_value(
                    'MODEL_DENOISE_STRENGTH',
                    inference_config,
                    'denoise_strength',
                    default=1.0,
                ),
                device=_get_value('DEVICE', inference_config, 'device', default='cuda:0'),
                tile=_get_int_value('MODEL_TILE', inference_config, 'tile', default=0),
                tile_pad=_get_int_value('MODEL_TILE_PAD', inference_config, 'tile_pad', default=10),
                pre_pad=_get_int_value('MODEL_PRE_PAD', inference_config, 'pre_pad', default=0),
                fp32=_get_bool_value('MODEL_FP32', inference_config, 'fp32', default=False),
                video_enabled=_get_bool_value('VIDEO_ENABLED', inference_config, 'video_enabled', default=True),
                ffmpeg_bin=_get_value('FFMPEG_BIN', inference_config, 'ffmpeg_bin', default='ffmpeg'),
                ffprobe_bin=_get_value('FFPROBE_BIN', inference_config, 'ffprobe_bin', default='ffprobe'),
                max_video_frames=_get_int_value('MAX_VIDEO_FRAMES', inference_config, 'max_video_frames', default=3000),
                max_video_seconds=_get_int_value('MAX_VIDEO_SECONDS', inference_config, 'max_video_seconds', default=180),
                video_processing_mode=_get_value(
                    'VIDEO_PROCESSING_MODE',
                    inference_config,
                    'video_processing_mode',
                    default='stream',
                ),
                video_frame_ext=_get_value('VIDEO_FRAME_EXT', inference_config, 'video_frame_ext', default='png'),
                video_codec=_get_value('VIDEO_CODEC', inference_config, 'video_codec', default='h264_nvenc'),
                video_codec_fallbacks=tuple(
                    _get_csv_values(
                        'VIDEO_CODEC_FALLBACKS',
                        inference_config,
                        'video_codec_fallbacks',
                        default='h264_mf,libx264,mpeg4',
                    ),
                ),
                video_pix_fmt=_get_value('VIDEO_PIX_FMT', inference_config, 'video_pix_fmt', default='yuv420p'),
                audio_fallback_no_audio=_get_bool_value(
                    'AUDIO_FALLBACK_NO_AUDIO',
                    inference_config,
                    'audio_fallback_no_audio',
                    default=True,
                ),
            ),
            runtime=RuntimeSettings(
                work_dir=_get_value('WORK_DIR', runtime_config, 'work_dir', default='./runtime'),
                worker_id=_get_value('WORKER_ID', runtime_config, 'worker_id', default=''),
                log_level=_get_value('LOG_LEVEL', runtime_config, 'log_level', default='INFO'),
            ),
        )


def _get_value(env_name: str, section: Dict[str, Any], key: str, required: bool = False, default: str = '') -> str:
    env_value = os.getenv(env_name, '').strip()
    if env_value:
        return env_value

    file_value = section.get(key)
    if isinstance(file_value, str) and file_value.strip():
        return file_value.strip()

    if file_value is not None and not isinstance(file_value, str):
        text_value = str(file_value).strip()
        if text_value:
            return text_value

    if required:
        raise ValueError(f'Missing required configuration: {env_name} (or {key} in config file)')

    return default


def _get_int_value(env_name: str, section: Dict[str, Any], key: str, default: int) -> int:
    value = _get_value(env_name, section, key, default=str(default))
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid integer configuration: {env_name}={value}') from exc


def _get_float_value(env_name: str, section: Dict[str, Any], key: str, default: float) -> float:
    value = _get_value(env_name, section, key, default=str(default))
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid float configuration: {env_name}={value}') from exc


def _get_bool_value(env_name: str, section: Dict[str, Any], key: str, default: bool) -> bool:
    env_value = os.getenv(env_name, '').strip()
    if env_value:
        return _parse_bool(env_name, env_value)

    file_value = section.get(key)
    if file_value is None:
        return default
    if isinstance(file_value, bool):
        return file_value
    if isinstance(file_value, str):
        return _parse_bool(env_name, file_value)
    raise ValueError(f'Invalid boolean configuration: {env_name}={file_value}')


def _parse_bool(name: str, value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ('1', 'true', 'yes', 'on'):
        return True
    if lowered in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'Invalid boolean configuration: {name}={value}')


def _get_csv_values(env_name: str, section: Dict[str, Any], key: str, default: str = '') -> list[str]:
    env_value = os.getenv(env_name, '').strip()
    raw_value: Any = env_value if env_value else section.get(key, default)
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(',') if part.strip()]


def _load_config_file(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}

    with open(path, 'r', encoding='utf-8') as file_obj:
        data = yaml.safe_load(file_obj) or {}

    if not isinstance(data, dict):
        raise ValueError(f'Invalid config file format: {path}')

    return data
