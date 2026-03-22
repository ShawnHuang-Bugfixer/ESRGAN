from queue import Queue
import threading

import pytest

from python_sr_service.config import InferenceSettings
from python_sr_service.domain.errors import ErrorCode, ServiceError
from python_sr_service.domain.schema import VideoOptions
import python_sr_service.pipeline.video_pipeline as video_pipeline_module
from python_sr_service.pipeline.video_pipeline import (
    VideoInferenceOutput,
    VideoPipeline,
    available_video_encoders,
    _QUEUE_EOF,
    _available_encoders,
    _build_video_inference_settings,
    _normalize_frame_ext,
    _parse_fps,
    _queue_put,
    _resolve_video_codecs,
)


def test_parse_fps_ratio_and_number():
    assert _parse_fps('24000/1001') == pytest.approx(23.976023976)
    assert _parse_fps('25') == pytest.approx(25.0)


def test_normalize_frame_ext_defaults_to_png():
    assert _normalize_frame_ext('') == 'png'
    assert _normalize_frame_ext('.JPG') == 'jpg'


def test_build_video_inference_settings_uses_video_model_and_tile():
    settings = InferenceSettings(
        model_name='RealESRGAN_x4plus',
        video_model_name='realesr-general-x4v3',
        video_model_weights='weights/realesr-general-x4v3.pth',
        tile=0,
        video_tile=512,
    )

    video_settings = _build_video_inference_settings(settings)

    assert video_settings.model_name == 'realesr-general-x4v3'
    assert video_settings.model_weights == 'weights/realesr-general-x4v3.pth'
    assert video_settings.tile == 512


def test_prepare_falls_back_to_image_model_when_video_weights_missing(monkeypatch):
    settings = InferenceSettings(
        model_name='RealESRGAN_x4plus',
        video_model_name='realesr-general-x4v3',
        video_tile=512,
        tile=0,
    )
    pipeline = VideoPipeline(settings)
    calls = []

    def fake_prepare(model_name_override=''):
        calls.append(model_name_override)
        if model_name_override == 'realesr-general-x4v3':
            raise ServiceError(
                code=ErrorCode.MODEL_NOT_FOUND,
                message='missing general weights',
                retryable=False,
            )

    monkeypatch.setattr(pipeline._image_pipeline, 'prepare', fake_prepare)
    monkeypatch.setattr(video_pipeline_module, 'ImagePipeline', lambda fallback_settings: pipeline._image_pipeline)

    pipeline.prepare()

    assert calls == ['realesr-general-x4v3', 'RealESRGAN_x4plus']
    assert pipeline._active_video_settings.model_name == 'RealESRGAN_x4plus'
    assert pipeline._active_video_settings.tile == 512


def test_validate_video_limits_by_frames():
    settings = InferenceSettings(max_video_frames=10, max_video_seconds=100)
    pipeline = VideoPipeline(settings)

    with pytest.raises(ServiceError) as exc_info:
        pipeline._validate_video_limits({
            'frame_count': 11,
            'duration_seconds': 1.0,
        })

    assert exc_info.value.code == ErrorCode.VIDEO_LIMIT_EXCEEDED


def test_validate_video_limits_by_duration():
    settings = InferenceSettings(max_video_frames=1000, max_video_seconds=3)
    pipeline = VideoPipeline(settings)

    with pytest.raises(ServiceError) as exc_info:
        pipeline._validate_video_limits({
            'frame_count': 10,
            'duration_seconds': 3.1,
        })

    assert exc_info.value.code == ErrorCode.VIDEO_LIMIT_EXCEEDED


def test_resolve_processing_mode_forces_stream_when_stream_only_enabled():
    pipeline = VideoPipeline(InferenceSettings(video_processing_mode='extract', video_stream_only=True))

    assert pipeline._resolve_processing_mode(VideoOptions(extract_frame_first=True)) == 'stream'
    assert pipeline._resolve_processing_mode(VideoOptions()) == 'stream'


def test_resolve_processing_mode_prefers_explicit_option_when_stream_only_disabled():
    pipeline = VideoPipeline(InferenceSettings(video_processing_mode='extract', video_stream_only=False))

    assert pipeline._resolve_processing_mode(VideoOptions(extract_frame_first=False)) == 'stream'
    assert pipeline._resolve_processing_mode(VideoOptions(extract_frame_first=True)) == 'extract'


def test_resolve_video_codecs_requires_primary_hardware_encoder(monkeypatch):
    settings = InferenceSettings(
        video_codec='h264_nvenc',
        video_codec_fallbacks=('libx264',),
        video_require_hw_encoder=True,
    )
    monkeypatch.setattr(video_pipeline_module, '_available_encoders', lambda _: {'h264_nvenc', 'libx264'})

    assert _resolve_video_codecs(settings, 'ffmpeg') == ['h264_nvenc']


def test_resolve_video_codecs_raises_when_required_hardware_encoder_missing(monkeypatch):
    settings = InferenceSettings(
        video_codec='h264_nvenc',
        video_codec_fallbacks=('libx264',),
        video_require_hw_encoder=True,
    )
    monkeypatch.setattr(video_pipeline_module, '_available_encoders', lambda _: {'libx264'})

    with pytest.raises(ServiceError) as exc_info:
        _resolve_video_codecs(settings, 'ffmpeg')

    assert exc_info.value.code == ErrorCode.FFMPEG_ERROR
    assert 'h264_nvenc' in str(exc_info.value)


def test_merge_video_fallback_when_hw_requirement_disabled(monkeypatch, tmp_path):
    settings = InferenceSettings(
        video_codec='libx264',
        video_codec_fallbacks=('mpeg4',),
        video_pix_fmt='yuv420p',
        video_require_hw_encoder=False,
    )
    pipeline = VideoPipeline(settings)

    calls = []

    def fake_run_command(cmd, error_code):
        calls.append(cmd)
        codec = cmd[cmd.index('-c:v') + 1]
        if codec == 'libx264':
            raise ServiceError(
                code=ErrorCode.FFMPEG_ERROR,
                message="Unknown encoder 'libx264'",
                retryable=True,
            )
        return ''

    monkeypatch.setattr(video_pipeline_module, '_run_command', fake_run_command)
    monkeypatch.setattr(video_pipeline_module, '_available_encoders', lambda _: set())

    output = tmp_path / 'out.mp4'
    pipeline._merge_video('frame%08d.png', 'input.mp4', str(output), fps=24.0, with_audio=False)

    codecs = [cmd[cmd.index('-c:v') + 1] for cmd in calls]
    assert codecs == ['libx264', 'mpeg4']


def test_run_does_not_fallback_to_extract_when_stream_only_enabled(monkeypatch, tmp_path):
    settings = InferenceSettings(video_processing_mode='stream', video_stream_only=True)
    pipeline = VideoPipeline(settings)
    output_file = tmp_path / 'out.mp4'

    monkeypatch.setattr(pipeline, '_probe_video', lambda _: {
        'fps': 24.0,
        'frame_count': 12,
        'duration_seconds': 0.5,
        'has_audio': False,
        'width': 16,
        'height': 16,
    })
    monkeypatch.setattr(pipeline, '_validate_video_limits', lambda meta: None)
    monkeypatch.setattr(video_pipeline_module.os.path, 'exists', lambda _: True)

    def fake_stream(*args, **kwargs):
        raise ServiceError(code=ErrorCode.FFMPEG_ERROR, message='pipe failed', retryable=True)

    def fake_extract(*args, **kwargs):
        raise AssertionError('extract fallback should stay disabled in speed-first mode')

    monkeypatch.setattr(pipeline, '_run_stream_mode', fake_stream)
    monkeypatch.setattr(pipeline, '_run_extract_mode', fake_extract)

    with pytest.raises(ServiceError) as exc_info:
        pipeline.run('input.mp4', str(output_file), scale=4, video_options=VideoOptions())

    assert exc_info.value.code == ErrorCode.FFMPEG_ERROR


def test_run_can_fallback_to_extract_when_stream_only_disabled(monkeypatch, tmp_path):
    settings = InferenceSettings(video_processing_mode='stream', video_stream_only=False, video_require_hw_encoder=False)
    pipeline = VideoPipeline(settings)
    output_file = tmp_path / 'out.mp4'

    monkeypatch.setattr(pipeline, '_probe_video', lambda _: {
        'fps': 24.0,
        'frame_count': 12,
        'duration_seconds': 0.5,
        'has_audio': False,
        'width': 16,
        'height': 16,
    })
    monkeypatch.setattr(pipeline, '_validate_video_limits', lambda meta: None)
    monkeypatch.setattr(video_pipeline_module.os.path, 'exists', lambda _: True)

    def fake_stream(*args, **kwargs):
        raise ServiceError(code=ErrorCode.FFMPEG_ERROR, message='pipe failed', retryable=True)

    def fake_extract(*args, **kwargs):
        return VideoInferenceOutput(
            output_path=str(output_file),
            cost_ms=12,
            frame_count=12,
            fps=24.0,
            had_audio=False,
            used_audio=False,
        )

    monkeypatch.setattr(pipeline, '_run_stream_mode', fake_stream)
    monkeypatch.setattr(pipeline, '_run_extract_mode', fake_extract)

    result = pipeline.run('input.mp4', str(output_file), scale=4, video_options=VideoOptions())

    assert result.output_path == str(output_file)
    assert result.frame_count == 12


def test_available_encoders_is_cached(monkeypatch):
    _available_encoders.cache_clear()
    calls = {'count': 0}

    class Result:
        stdout = ' V..... h264_nvenc\n V..... libx264\n'

    def fake_run(*args, **kwargs):
        calls['count'] += 1
        return Result()

    monkeypatch.setattr(video_pipeline_module.subprocess, 'run', fake_run)

    first = _available_encoders('ffmpeg')
    second = _available_encoders('ffmpeg')

    assert first == {'h264_nvenc', 'libx264'}
    assert second == first
    assert calls['count'] == 1


def test_queue_put_returns_when_stop_event_is_set_and_queue_is_full():
    queue_obj = Queue(maxsize=1)
    queue_obj.put('busy')
    stop_event = threading.Event()
    stop_event.set()

    _queue_put(queue_obj, _QUEUE_EOF, stop_event)

    assert queue_obj.qsize() == 1


def test_resolve_video_codecs_raises_when_no_configured_codec_is_available(monkeypatch):
    settings = InferenceSettings(
        video_codec='h264_nvenc',
        video_codec_fallbacks=('libx264',),
        video_require_hw_encoder=False,
    )
    monkeypatch.setattr(video_pipeline_module, '_available_encoders', lambda _: {'h264_mf', 'mpeg4'})

    with pytest.raises(ServiceError) as exc_info:
        _resolve_video_codecs(settings, 'ffmpeg')

    assert exc_info.value.retryable is False
    assert 'No usable configured video encoder' in str(exc_info.value)


def test_available_video_encoders_returns_sorted_values(monkeypatch):
    monkeypatch.setattr(video_pipeline_module, '_available_encoders', lambda _: {'mpeg4', 'h264_mf'})

    assert available_video_encoders('ffmpeg') == ['h264_mf', 'mpeg4']
