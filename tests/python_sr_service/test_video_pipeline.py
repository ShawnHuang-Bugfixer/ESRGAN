import pytest

from python_sr_service.config import InferenceSettings
from python_sr_service.domain.errors import ErrorCode, ServiceError
from python_sr_service.domain.schema import VideoOptions
import python_sr_service.pipeline.video_pipeline as video_pipeline_module
from python_sr_service.pipeline.video_pipeline import VideoInferenceOutput, VideoPipeline, _available_encoders, _normalize_frame_ext, _parse_fps


def test_parse_fps_ratio_and_number():
    assert _parse_fps('24000/1001') == pytest.approx(23.976023976)
    assert _parse_fps('25') == pytest.approx(25.0)


def test_normalize_frame_ext_defaults_to_png():
    assert _normalize_frame_ext('') == 'png'
    assert _normalize_frame_ext('.JPG') == 'jpg'


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


def test_resolve_processing_mode_prefers_explicit_option():
    pipeline = VideoPipeline(InferenceSettings(video_processing_mode='extract'))

    assert pipeline._resolve_processing_mode(VideoOptions(extract_frame_first=False)) == 'stream'
    assert pipeline._resolve_processing_mode(VideoOptions(extract_frame_first=True)) == 'extract'


def test_resolve_processing_mode_uses_config_default():
    pipeline = VideoPipeline(InferenceSettings(video_processing_mode='extract'))

    assert pipeline._resolve_processing_mode(VideoOptions()) == 'extract'


def test_merge_video_fallback_when_primary_encoder_unavailable(monkeypatch, tmp_path):
    settings = InferenceSettings(video_codec='libx264', video_codec_fallbacks=('mpeg4',), video_pix_fmt='yuv420p')
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


def test_merge_video_raises_on_non_encoder_error(monkeypatch, tmp_path):
    settings = InferenceSettings(video_codec='libx264', video_codec_fallbacks=('mpeg4',), video_pix_fmt='yuv420p')
    pipeline = VideoPipeline(settings)

    def fake_run_command(_, __):
        raise ServiceError(
            code=ErrorCode.FFMPEG_ERROR,
            message='Permission denied while writing output file',
            retryable=True,
        )

    monkeypatch.setattr(video_pipeline_module, '_run_command', fake_run_command)
    monkeypatch.setattr(video_pipeline_module, '_available_encoders', lambda _: set())

    output = tmp_path / 'out.mp4'
    with pytest.raises(ServiceError) as exc_info:
        pipeline._merge_video('frame%08d.png', 'input.mp4', str(output), fps=24.0, with_audio=False)

    assert 'Permission denied' in str(exc_info.value)


def test_run_falls_back_to_extract_when_stream_mode_hits_ffmpeg_error(monkeypatch, tmp_path):
    settings = InferenceSettings(video_processing_mode='stream')
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
    monkeypatch.setattr(video_pipeline_module.os.path, 'exists', lambda _: True)

    result = pipeline.run('input.mp4', str(output_file), scale=4, video_options=VideoOptions())

    assert result.output_path == str(output_file)
    assert result.frame_count == 12
    assert result.used_audio is False


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
