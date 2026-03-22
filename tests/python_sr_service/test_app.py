import logging

import pytest

import python_sr_service.app as app_module
from python_sr_service.app import _validate_video_runtime
from python_sr_service.domain.errors import ErrorCode, ServiceError
from python_sr_service.config import (
    COSSettings,
    IdempotencySettings,
    InferenceSettings,
    MySQLSettings,
    RabbitMQSettings,
    RedisSettings,
    RuntimeSettings,
    Settings,
)


def _build_settings(inference: InferenceSettings) -> Settings:
    return Settings(
        cos=COSSettings(secret_id='id', secret_key='key', region='ap-chengdu', bucket='bucket'),
        mysql=MySQLSettings(),
        rabbitmq=RabbitMQSettings(url='amqp://guest:guest@localhost:5672/%2F'),
        redis=RedisSettings(url='redis://localhost:6379/0'),
        idempotency=IdempotencySettings(),
        inference=inference,
        runtime=RuntimeSettings(),
    )


def test_validate_video_runtime_raises_when_binaries_missing():
    settings = _build_settings(
        InferenceSettings(
            video_enabled=True,
            ffmpeg_bin='__missing_ffmpeg__',
            ffprobe_bin='__missing_ffprobe__',
        )
    )

    with pytest.raises(RuntimeError) as exc_info:
        _validate_video_runtime(logging.getLogger(__name__), settings)

    assert 'ffmpeg_bin' in str(exc_info.value)
    assert 'ffprobe_bin' in str(exc_info.value)


def test_validate_video_runtime_skips_when_video_disabled():
    settings = _build_settings(
        InferenceSettings(
            video_enabled=False,
            ffmpeg_bin='__missing_ffmpeg__',
            ffprobe_bin='__missing_ffprobe__',
        )
    )

    _validate_video_runtime(logging.getLogger(__name__), settings)


def test_validate_video_runtime_checks_video_codecs(tmp_path, monkeypatch):
    ffmpeg_bin = tmp_path / 'ffmpeg.exe'
    ffprobe_bin = tmp_path / 'ffprobe.exe'
    ffmpeg_bin.write_text('', encoding='utf-8')
    ffprobe_bin.write_text('', encoding='utf-8')
    settings = _build_settings(
        InferenceSettings(
            video_enabled=True,
            ffmpeg_bin=str(ffmpeg_bin),
            ffprobe_bin=str(ffprobe_bin),
            video_codec='h264_mf',
            video_codec_fallbacks=('mpeg4',),
            video_require_hw_encoder=False,
        )
    )

    monkeypatch.setattr(app_module, 'available_video_encoders', lambda _: ['h264_mf', 'mpeg4'])
    monkeypatch.setattr(app_module, 'resolve_video_codec_chain', lambda _: ['h264_mf', 'mpeg4'])

    _validate_video_runtime(logging.getLogger(__name__), settings)


def test_validate_video_runtime_raises_when_configured_codecs_unavailable(tmp_path, monkeypatch):
    ffmpeg_bin = tmp_path / 'ffmpeg.exe'
    ffprobe_bin = tmp_path / 'ffprobe.exe'
    ffmpeg_bin.write_text('', encoding='utf-8')
    ffprobe_bin.write_text('', encoding='utf-8')
    settings = _build_settings(
        InferenceSettings(
            video_enabled=True,
            ffmpeg_bin=str(ffmpeg_bin),
            ffprobe_bin=str(ffprobe_bin),
            video_codec='h264_nvenc',
            video_require_hw_encoder=True,
        )
    )

    monkeypatch.setattr(app_module, 'available_video_encoders', lambda _: ['h264_mf', 'mpeg4'])
    monkeypatch.setattr(
        app_module,
        'resolve_video_codec_chain',
        lambda _: (_ for _ in ()).throw(
            ServiceError(
                code=ErrorCode.FFMPEG_ERROR,
                message='Required hardware video encoder unavailable: h264_nvenc',
                retryable=False,
            )
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        _validate_video_runtime(logging.getLogger(__name__), settings)

    assert 'h264_nvenc' in str(exc_info.value)
