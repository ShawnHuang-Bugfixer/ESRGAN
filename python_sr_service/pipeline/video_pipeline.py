from dataclasses import dataclass, replace
from functools import lru_cache
import glob
import json
import logging
import os
from queue import Empty, Full, Queue
import shutil
import subprocess
import threading
import time
from fractions import Fraction
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np

from python_sr_service.config import InferenceSettings
from python_sr_service.domain.errors import ErrorCode, ServiceError
from python_sr_service.domain.schema import VideoOptions
from python_sr_service.pipeline.image_pipeline import ImagePipeline, InferenceOutput
from python_sr_service.runtime.logging import format_log_fields

logger = logging.getLogger(__name__)

_QUEUE_EOF = object()
_STREAM_QUEUE_SIZE = 4
_STREAM_HEARTBEAT_FRAMES = 10


@dataclass(frozen=True)
class VideoInferenceOutput(InferenceOutput):
    frame_count: int
    fps: float
    had_audio: bool
    used_audio: bool


class VideoPipeline:
    def __init__(self, settings: InferenceSettings, image_pipeline: Optional[ImagePipeline] = None):
        self._settings = settings
        self._video_settings = _build_video_inference_settings(settings)
        self._active_video_settings = self._video_settings
        self._image_pipeline = image_pipeline or ImagePipeline(self._active_video_settings)

    def prepare(self) -> None:
        try:
            self._image_pipeline.prepare(model_name_override=self._video_settings.model_name)
            self._active_video_settings = self._video_settings
        except ServiceError as exc:
            if exc.code != ErrorCode.MODEL_NOT_FOUND:
                raise
            fallback_settings = _build_video_fallback_settings(self._settings)
            logger.warning(
                'video_model_fallback %s',
                format_log_fields(
                    {
                        'requestedModel': self._video_settings.model_name,
                        'fallbackModel': fallback_settings.model_name,
                        'reason': str(exc),
                    },
                ),
            )
            self._active_video_settings = fallback_settings
            self._image_pipeline = ImagePipeline(self._active_video_settings)
            self._image_pipeline.prepare(model_name_override=self._active_video_settings.model_name)

    @property
    def active_model_name(self) -> str:
        return self._active_video_settings.model_name.strip()

    @property
    def active_tile(self) -> int:
        return int(self._active_video_settings.tile)

    def run(
        self,
        input_file: str,
        output_file: str,
        scale: int,
        model_name_override: str = '',
        video_options: Optional[VideoOptions] = None,
        phase_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> VideoInferenceOutput:
        options = video_options or VideoOptions()
        started = time.time()

        if not os.path.exists(input_file):
            raise ServiceError(
                code=ErrorCode.INPUT_NOT_FOUND,
                message=f'Input video not found: {input_file}',
                retryable=False,
            )

        meta = self._probe_video(input_file)
        self._validate_video_limits(meta)
        _emit(phase_callback, 'video_probed', {
            'fps': meta['fps'],
            'frameCount': meta['frame_count'],
            'durationSeconds': meta['duration_seconds'],
            'hasAudio': meta['has_audio'],
            'width': meta['width'],
            'height': meta['height'],
        })

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        fps = options.fps_override if options.fps_override is not None else meta['fps']
        mode = self._resolve_processing_mode(options)
        model_name = (model_name_override or self._active_video_settings.model_name).strip()
        logger.info(
            'video_processing_mode_selected %s',
            format_log_fields(
                {
                    'inputFile': input_file,
                    'outputFile': output_file,
                    'mode': mode,
                    'fps': fps,
                    'frameCount': meta['frame_count'],
                    'modelName': model_name,
                    'tile': self._active_video_settings.tile,
                },
            ),
        )

        if mode == 'extract':
            result = self._run_extract_mode(
                input_file,
                output_file,
                scale,
                fps,
                meta,
                model_name,
                options,
                phase_callback,
            )
        else:
            try:
                result = self._run_stream_mode(
                    input_file,
                    output_file,
                    scale,
                    fps,
                    meta,
                    model_name,
                    options,
                    phase_callback,
                )
            except ServiceError as exc:
                if not self._settings.video_stream_only and _should_fallback_to_extract(exc):
                    logger.warning(
                        'video_stream_fallback_extract %s',
                        format_log_fields(
                            {
                                'inputFile': input_file,
                                'outputFile': output_file,
                                'errorCode': exc.code.value,
                                'errorMsg': str(exc),
                            },
                        ),
                    )
                    result = self._run_extract_mode(
                        input_file,
                        output_file,
                        scale,
                        fps,
                        meta,
                        model_name,
                        options,
                        phase_callback,
                    )
                else:
                    raise

        return VideoInferenceOutput(
            output_path=output_file,
            cost_ms=max(result.cost_ms, int((time.time() - started) * 1000)),
            frame_count=result.frame_count,
            fps=result.fps,
            had_audio=meta['has_audio'],
            used_audio=result.used_audio,
        )

    def _probe_video(self, input_file: str) -> Dict[str, Any]:
        cmd = [
            self._settings.ffprobe_bin,
            '-v',
            'error',
            '-print_format',
            'json',
            '-show_streams',
            '-show_format',
            input_file,
        ]
        output = _run_command(cmd, ErrorCode.FFMPEG_ERROR)
        try:
            data = json.loads(output)
        except json.JSONDecodeError as exc:
            raise ServiceError(
                code=ErrorCode.FFMPEG_ERROR,
                message='Failed to parse ffprobe output',
                retryable=True,
                cause=exc,
            ) from exc

        streams = data.get('streams') or []
        video_stream = next((stream for stream in streams if stream.get('codec_type') == 'video'), None)
        if not video_stream:
            raise ServiceError(
                code=ErrorCode.INPUT_INVALID,
                message='No video stream found in input file',
                retryable=False,
            )

        has_audio = any(stream.get('codec_type') == 'audio' for stream in streams)
        fps_text = video_stream.get('avg_frame_rate') or video_stream.get('r_frame_rate') or '0/1'
        fps = _parse_fps(fps_text)

        duration_seconds = _to_float(
            video_stream.get('duration') or (data.get('format') or {}).get('duration'),
            default=0.0,
        )
        frame_count = _to_int(video_stream.get('nb_frames'), default=0)
        if frame_count <= 0 and duration_seconds > 0 and fps > 0:
            frame_count = int(round(duration_seconds * fps))

        return {
            'fps': fps,
            'frame_count': frame_count,
            'duration_seconds': duration_seconds,
            'has_audio': has_audio,
            'width': _to_int(video_stream.get('width'), default=0),
            'height': _to_int(video_stream.get('height'), default=0),
        }

    def _validate_video_limits(self, meta: Dict[str, Any]) -> None:
        frame_count = int(meta['frame_count'])
        duration_seconds = float(meta['duration_seconds'])

        if frame_count > self._settings.max_video_frames:
            raise ServiceError(
                code=ErrorCode.VIDEO_LIMIT_EXCEEDED,
                message=(
                    f'Video frame count {frame_count} exceeds the limit '
                    f'{self._settings.max_video_frames}'
                ),
                retryable=False,
            )

        if duration_seconds > self._settings.max_video_seconds:
            raise ServiceError(
                code=ErrorCode.VIDEO_LIMIT_EXCEEDED,
                message=(
                    f'Video duration {duration_seconds:.2f}s exceeds the limit '
                    f'{self._settings.max_video_seconds}s'
                ),
                retryable=False,
            )

    def _resolve_processing_mode(self, options: VideoOptions) -> str:
        if self._settings.video_stream_only:
            if options.extract_frame_first is True:
                logger.warning('video_extract_request_ignored %s', format_log_fields({'forcedMode': 'stream'}))
            return 'stream'

        if options.extract_frame_first is True:
            return 'extract'
        if options.extract_frame_first is False:
            return 'stream'

        configured = str(self._settings.video_processing_mode or 'stream').strip().lower()
        if configured in ('stream', 'extract'):
            return configured
        logger.warning('video_processing_mode_invalid %s', format_log_fields({'configured': configured, 'fallback': 'stream'}))
        return 'stream'

    def _run_extract_mode(
        self,
        input_file: str,
        output_file: str,
        scale: int,
        fps: float,
        meta: Dict[str, Any],
        model_name_override: str,
        options: VideoOptions,
        phase_callback: Optional[Callable[[str, Dict[str, Any]], None]],
    ) -> VideoInferenceOutput:
        started = time.time()
        frame_ext = _normalize_frame_ext(self._settings.video_frame_ext)
        frame_input_dir = os.path.join(os.path.dirname(output_file), '_frames_in')
        frame_output_dir = os.path.join(os.path.dirname(output_file), '_frames_out')
        os.makedirs(frame_input_dir, exist_ok=True)
        os.makedirs(frame_output_dir, exist_ok=True)
        try:
            frame_pattern = os.path.join(frame_input_dir, f'frame%08d.{frame_ext}')
            self._extract_frames(input_file, frame_pattern)
            frame_paths = sorted(glob.glob(os.path.join(frame_input_dir, f'*.{frame_ext}')))
            if not frame_paths:
                raise ServiceError(
                    code=ErrorCode.INPUT_INVALID,
                    message='No frames extracted from input video',
                    retryable=False,
                )

            _emit(phase_callback, 'frames_extracted', {'frameCount': len(frame_paths), 'frameExt': frame_ext, 'mode': 'extract'})

            for idx, frame_path in enumerate(frame_paths, start=1):
                image = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ServiceError(
                        code=ErrorCode.INPUT_INVALID,
                        message=f'Failed to read extracted frame: {frame_path}',
                        retryable=False,
                    )

                output, _ = self._image_pipeline.enhance_array(image, scale, model_name_override=model_name_override)
                output_path = os.path.join(frame_output_dir, os.path.basename(frame_path))
                if not cv2.imwrite(output_path, output):
                    raise ServiceError(
                        code=ErrorCode.INFER_RUNTIME_ERROR,
                        message=f'Failed to write enhanced frame: {output_path}',
                        retryable=False,
                    )

                _emit(phase_callback, 'frame_enhanced', {'frameIndex': idx, 'totalFrames': len(frame_paths)})

            _emit(phase_callback, 'frames_inferred', {'frameCount': len(frame_paths), 'mode': 'extract'})

            pattern_for_merge = os.path.join(frame_output_dir, f'frame%08d.{frame_ext}')
            merge_with_audio = bool(options.keep_audio and meta['has_audio'])
            used_audio = merge_with_audio

            if merge_with_audio:
                try:
                    self._merge_video(pattern_for_merge, input_file, output_file, fps, with_audio=True)
                except ServiceError:
                    if self._settings.audio_fallback_no_audio:
                        logger.warning(
                            'video_merge_audio_failed %s',
                            format_log_fields({'outputFile': output_file, 'fallback': 'no-audio', 'mode': 'extract'}),
                        )
                        _emit(phase_callback, 'audio_fallback', {'enabled': True, 'mode': 'extract'})
                        used_audio = False
                        self._merge_video(pattern_for_merge, input_file, output_file, fps, with_audio=False)
                    else:
                        raise
            else:
                self._merge_video(pattern_for_merge, input_file, output_file, fps, with_audio=False)

            _emit(phase_callback, 'video_merged', {'usedAudio': used_audio, 'fps': fps, 'merged': True, 'mode': 'extract'})

            return VideoInferenceOutput(
                output_path=output_file,
                cost_ms=int((time.time() - started) * 1000),
                frame_count=len(frame_paths),
                fps=fps,
                had_audio=meta['has_audio'],
                used_audio=used_audio,
            )
        finally:
            shutil.rmtree(frame_input_dir, ignore_errors=True)
            shutil.rmtree(frame_output_dir, ignore_errors=True)

    def _run_stream_mode(
        self,
        input_file: str,
        output_file: str,
        scale: int,
        fps: float,
        meta: Dict[str, Any],
        model_name_override: str,
        options: VideoOptions,
        phase_callback: Optional[Callable[[str, Dict[str, Any]], None]],
    ) -> VideoInferenceOutput:
        merge_with_audio = bool(options.keep_audio and meta['has_audio'])
        try:
            return self._stream_once(
                input_file,
                output_file,
                scale,
                fps,
                meta,
                model_name_override,
                phase_callback,
                with_audio=merge_with_audio,
            )
        except ServiceError as exc:
            if merge_with_audio and self._settings.audio_fallback_no_audio:
                logger.warning(
                    'video_stream_audio_failed %s',
                    format_log_fields({'outputFile': output_file, 'fallback': 'no-audio', 'errorMsg': str(exc)}),
                )
                _emit(phase_callback, 'audio_fallback', {'enabled': True, 'mode': 'stream'})
                return self._stream_once(
                    input_file,
                    output_file,
                    scale,
                    fps,
                    meta,
                    model_name_override,
                    phase_callback,
                    with_audio=False,
                )
            raise

    def _stream_once(
        self,
        input_file: str,
        output_file: str,
        scale: int,
        fps: float,
        meta: Dict[str, Any],
        model_name_override: str,
        phase_callback: Optional[Callable[[str, Dict[str, Any]], None]],
        with_audio: bool,
    ) -> VideoInferenceOutput:
        started = time.time()
        width = int(meta['width'])
        height = int(meta['height'])
        if width <= 0 or height <= 0:
            raise ServiceError(
                code=ErrorCode.INPUT_INVALID,
                message='Invalid video resolution from ffprobe',
                retryable=False,
            )

        out_width = max(1, int(round(width * scale)))
        out_height = max(1, int(round(height * scale)))
        frame_size = width * height * 3
        expected_frames = max(int(meta['frame_count']), 0)
        _emit(phase_callback, 'frames_extracted', {'frameCount': expected_frames, 'frameExt': 'rawvideo', 'mode': 'stream'})
        logger.info(
            'video_stream_start %s',
            format_log_fields(
                {
                    'inputFile': input_file,
                    'outputFile': output_file,
                    'width': width,
                    'height': height,
                    'outWidth': out_width,
                    'outHeight': out_height,
                    'fps': fps,
                    'frameCount': expected_frames,
                    'modelName': model_name_override,
                    'tile': self._active_video_settings.tile,
                    'withAudio': with_audio,
                },
            ),
        )

        codecs = _resolve_video_codecs(self._settings, self._settings.ffmpeg_bin)
        last_error: Optional[ServiceError] = None

        for codec in codecs:
            decoder = None
            writer = None
            decode_queue: Queue[Any] = Queue(maxsize=_STREAM_QUEUE_SIZE)
            encode_queue: Queue[Any] = Queue(maxsize=_STREAM_QUEUE_SIZE)
            stop_event = threading.Event()
            errors: list[ServiceError] = []
            processed_frames = {'value': 0}
            decoded_frames = {'value': 0}
            inferred_frames = {'value': 0}
            failed_stage = {'value': ''}

            def record_error(stage: str, exc: ServiceError) -> None:
                failed_stage['value'] = stage
                if not errors:
                    errors.append(exc)
                stop_event.set()

            try:
                decoder = subprocess.Popen(
                    _decoder_command(self._settings.ffmpeg_bin, input_file),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                writer = subprocess.Popen(
                    _writer_command(
                        self._settings.ffmpeg_bin,
                        output_file,
                        input_file,
                        fps,
                        out_width,
                        out_height,
                        codec,
                        self._settings.video_pix_fmt,
                        with_audio,
                    ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if decoder.stdout is None or writer.stdin is None:
                    raise ServiceError(
                        code=ErrorCode.FFMPEG_ERROR,
                        message='Failed to open ffmpeg pipes for streaming',
                        retryable=True,
                    )

                def decode_worker() -> None:
                    frame_index = 0
                    logger.info('video_decode_worker_start %s', format_log_fields({'codec': codec, 'inputFile': input_file}))
                    try:
                        while not stop_event.is_set():
                            frame_bytes = _read_exact(decoder.stdout, frame_size)
                            if not frame_bytes:
                                break
                            if len(frame_bytes) != frame_size:
                                raise ServiceError(
                                    code=ErrorCode.FFMPEG_ERROR,
                                    message='Unexpected EOF while reading decoded video frames',
                                    retryable=True,
                                )
                            frame_index += 1
                            decoded_frames['value'] = frame_index
                            image = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3)).copy()
                            _queue_put(decode_queue, (frame_index, image), stop_event)
                    except ServiceError as exc:
                        record_error('decode', exc)
                    except Exception as exc:
                        record_error(
                            'decode',
                            ServiceError(
                                code=ErrorCode.FFMPEG_ERROR,
                                message=str(exc),
                                retryable=True,
                                cause=exc,
                            ),
                        )
                    finally:
                        logger.info(
                            'video_decode_worker_done %s',
                            format_log_fields({'codec': codec, 'decodedFrames': decoded_frames['value'], 'stopped': stop_event.is_set()}),
                        )
                        _queue_put(decode_queue, _QUEUE_EOF, stop_event)

                def infer_worker() -> None:
                    logger.info('video_infer_worker_start %s', format_log_fields({'modelName': model_name_override, 'tile': self._active_video_settings.tile}))
                    try:
                        while not stop_event.is_set():
                            item = _queue_get(decode_queue, stop_event)
                            if item is _QUEUE_EOF:
                                break
                            frame_index, image = item
                            output, _ = self._image_pipeline.enhance_array(
                                image,
                                scale,
                                model_name_override=model_name_override,
                            )
                            inferred_frames['value'] = frame_index
                            _queue_put(encode_queue, (frame_index, np.ascontiguousarray(output)), stop_event)
                    except ServiceError as exc:
                        record_error('infer', exc)
                    except Exception as exc:
                        record_error(
                            'infer',
                            ServiceError(
                                code=ErrorCode.INFER_RUNTIME_ERROR,
                                message='Failed to run video frame inference',
                                retryable=True,
                                cause=exc,
                            ),
                        )
                    finally:
                        logger.info(
                            'video_infer_worker_done %s',
                            format_log_fields({'modelName': model_name_override, 'inferredFrames': inferred_frames['value'], 'stopped': stop_event.is_set()}),
                        )
                        _queue_put(encode_queue, _QUEUE_EOF, stop_event)

                def encode_worker() -> None:
                    logger.info('video_encode_worker_start %s', format_log_fields({'codec': codec, 'outputFile': output_file, 'withAudio': with_audio}))
                    try:
                        while not stop_event.is_set():
                            item = _queue_get(encode_queue, stop_event)
                            if item is _QUEUE_EOF:
                                break
                            frame_index, output = item
                            try:
                                writer.stdin.write(output.tobytes())
                            except (BrokenPipeError, OSError) as exc:
                                stderr = _drain_pipe(writer.stderr)
                                raise ServiceError(
                                    code=ErrorCode.FFMPEG_ERROR,
                                    message=stderr or str(exc),
                                    retryable=True,
                                    cause=exc,
                                ) from exc
                            processed_frames['value'] = frame_index
                            elapsed_ms = max(int((time.time() - started) * 1000), 1)
                            avg_fps = round(frame_index * 1000 / elapsed_ms, 2)
                            _emit(
                                phase_callback,
                                'frame_enhanced',
                                {
                                    'frameIndex': frame_index,
                                    'totalFrames': max(expected_frames, frame_index),
                                    'elapsedMs': elapsed_ms,
                                    'avgFps': avg_fps,
                                    'decodeQueueSize': decode_queue.qsize(),
                                    'encodeQueueSize': encode_queue.qsize(),
                                    'codec': codec,
                                    'withAudio': with_audio,
                                    'modelName': model_name_override,
                                    'tile': self._active_video_settings.tile,
                                },
                            )
                    except ServiceError as exc:
                        record_error('encode', exc)
                    except Exception as exc:
                        record_error(
                            'encode',
                            ServiceError(
                                code=ErrorCode.FFMPEG_ERROR,
                                message=str(exc),
                                retryable=True,
                                cause=exc,
                            ),
                        )
                    finally:
                        logger.info(
                            'video_encode_worker_done %s',
                            format_log_fields({'codec': codec, 'encodedFrames': processed_frames['value'], 'stopped': stop_event.is_set()}),
                        )
                        _close_stdin(writer)

                threads = [
                    threading.Thread(target=decode_worker, name='video-decode', daemon=True),
                    threading.Thread(target=infer_worker, name='video-infer', daemon=True),
                    threading.Thread(target=encode_worker, name='video-encode', daemon=True),
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                if errors:
                    raise errors[0]

                decoder_stderr = _drain_pipe(decoder.stderr)
                writer_stderr = _drain_pipe(writer.stderr)
                decoder_rc = decoder.wait()
                writer_rc = writer.wait()

                if processed_frames['value'] <= 0:
                    raise ServiceError(
                        code=ErrorCode.INPUT_INVALID,
                        message='No frames decoded from input video',
                        retryable=False,
                    )
                if decoder_rc != 0:
                    raise ServiceError(
                        code=ErrorCode.FFMPEG_ERROR,
                        message=decoder_stderr or 'ffmpeg decoder failed',
                        retryable=True,
                    )
                if writer_rc != 0:
                    raise ServiceError(
                        code=ErrorCode.FFMPEG_ERROR,
                        message=writer_stderr or 'ffmpeg writer failed',
                        retryable=True,
                    )

                total_cost_ms = int((time.time() - started) * 1000)
                avg_fps = round(processed_frames['value'] * 1000 / max(total_cost_ms, 1), 2)
                _emit(phase_callback, 'frames_inferred', {'frameCount': processed_frames['value'], 'mode': 'stream'})
                _emit(phase_callback, 'video_merged', {'usedAudio': with_audio, 'fps': fps, 'merged': True, 'mode': 'stream'})
                logger.info(
                    'video_stream_done %s',
                    format_log_fields(
                        {
                            'codec': codec,
                            'frameCount': processed_frames['value'],
                            'costMs': total_cost_ms,
                            'avgFps': avg_fps,
                            'withAudio': with_audio,
                            'modelName': model_name_override,
                            'tile': self._active_video_settings.tile,
                        },
                    ),
                )
                if codec != self._settings.video_codec:
                    logger.warning(
                        'video_codec_fallback_used %s',
                        format_log_fields({'fromCodec': self._settings.video_codec, 'toCodec': codec}),
                    )

                return VideoInferenceOutput(
                    output_path=output_file,
                    cost_ms=total_cost_ms,
                    frame_count=processed_frames['value'],
                    fps=fps,
                    had_audio=meta['has_audio'],
                    used_audio=with_audio,
                )
            except ServiceError as exc:
                last_error = exc
                logger.error(
                    'video_stream_failed %s',
                    format_log_fields(
                        {
                            'codec': codec,
                            'failedStage': failed_stage['value'] or 'stream',
                            'processedFrames': processed_frames['value'],
                            'decodedFrames': decoded_frames['value'],
                            'inferredFrames': inferred_frames['value'],
                            'errorCode': exc.code.value,
                            'errorMsg': str(exc),
                        },
                    ),
                )
                if _is_unknown_encoder_error(str(exc)):
                    logger.warning('video_codec_unavailable %s', format_log_fields({'codec': codec}))
                    continue
                raise
            finally:
                stop_event.set()
                _terminate_process(decoder)
                _terminate_process(writer)

        if last_error is not None:
            raise last_error
        raise ServiceError(
            code=ErrorCode.FFMPEG_ERROR,
            message='No usable video encoder is available',
            retryable=True,
        )

    def _extract_frames(self, input_file: str, frame_pattern: str) -> None:
        cmd = [
            self._settings.ffmpeg_bin,
            '-y',
            '-i',
            input_file,
            '-vsync',
            '0',
            frame_pattern,
        ]
        _run_command(cmd, ErrorCode.FFMPEG_ERROR)

    def _merge_video(self, frame_pattern: str, input_file: str, output_file: str, fps: float, with_audio: bool) -> None:
        codecs = _resolve_video_codecs(self._settings, self._settings.ffmpeg_bin)
        last_error: Optional[ServiceError] = None

        for codec in codecs:
            cmd = [
                self._settings.ffmpeg_bin,
                '-y',
                '-framerate',
                str(fps),
                '-i',
                frame_pattern,
            ]

            if with_audio:
                cmd.extend([
                    '-i',
                    input_file,
                    '-map',
                    '0:v:0',
                    '-map',
                    '1:a:0',
                    '-c:a',
                    'copy',
                ])

            cmd.extend([
                '-c:v',
                codec,
                '-r',
                str(fps),
                '-pix_fmt',
                self._settings.video_pix_fmt,
                output_file,
            ])

            try:
                _run_command(cmd, ErrorCode.FFMPEG_ERROR)
                if codec != self._settings.video_codec:
                    logger.warning(
                        'video_codec_fallback_used %s',
                        format_log_fields({'fromCodec': self._settings.video_codec, 'toCodec': codec}),
                    )
                return
            except ServiceError as exc:
                last_error = exc
                if _is_unknown_encoder_error(str(exc)):
                    logger.warning('video_codec_unavailable %s', format_log_fields({'codec': codec}))
                    continue
                raise

        if last_error is not None:
            raise last_error


def _build_video_inference_settings(settings: InferenceSettings) -> InferenceSettings:
    video_model_name = settings.video_model_name.strip() or settings.model_name.strip()
    return replace(
        settings,
        model_name=video_model_name,
        model_weights=settings.video_model_weights.strip(),
        tile=max(0, int(settings.video_tile)),
    )


def _build_video_fallback_settings(settings: InferenceSettings) -> InferenceSettings:
    fallback_model_name = settings.model_name.strip()
    return replace(
        settings,
        model_name=fallback_model_name,
        model_weights=settings.model_weights.strip(),
        tile=max(0, int(settings.video_tile)),
    )


def available_video_encoders(ffmpeg_bin: str) -> list[str]:
    return sorted(_available_encoders(ffmpeg_bin))


def resolve_video_codec_chain(settings: InferenceSettings) -> list[str]:
    return _resolve_video_codecs(settings, settings.ffmpeg_bin)


def _resolve_video_codecs(settings: InferenceSettings, ffmpeg_bin: str) -> list[str]:
    configured_codecs = _codec_candidates(settings.video_codec, settings.video_codec_fallbacks)
    available = _available_encoders(ffmpeg_bin)
    codecs = [codec for codec in configured_codecs if codec in available] if available else configured_codecs
    if settings.video_require_hw_encoder:
        primary = settings.video_codec.strip()
        if primary not in codecs:
            raise ServiceError(
                code=ErrorCode.FFMPEG_ERROR,
                message=f'Required hardware video encoder unavailable: {primary}',
                retryable=False,
            )
        return [primary]
    if available and not codecs:
        raise ServiceError(
            code=ErrorCode.FFMPEG_ERROR,
            message=f'No usable configured video encoder is available: {configured_codecs}',
            retryable=False,
        )
    return codecs


def _run_command(command: list[str], error_code: ErrorCode) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or 'subprocess failed'
        raise ServiceError(
            code=error_code,
            message=message,
            retryable=True,
            cause=exc,
        ) from exc
    except OSError as exc:
        raise ServiceError(
            code=error_code,
            message=str(exc),
            retryable=True,
            cause=exc,
        ) from exc


def _parse_fps(value: str) -> float:
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        if '/' in text:
            ratio = Fraction(text)
            return float(ratio)
        return float(text)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_frame_ext(value: str) -> str:
    text = str(value or 'png').strip().lower().lstrip('.')
    return text or 'png'


def _emit(callback: Optional[Callable[[str, Dict[str, Any]], None]], phase: str, payload: Dict[str, Any]) -> None:
    if callback is not None:
        callback(phase, payload)


def _codec_candidates(primary: str, fallbacks: tuple[str, ...]) -> list[str]:
    candidates: list[str] = []
    for codec in [primary, *fallbacks]:
        text = str(codec).strip()
        if not text or text in candidates:
            continue
        candidates.append(text)
    return candidates or ['mpeg4']


def _is_unknown_encoder_error(message: str) -> bool:
    lowered = str(message).lower()
    return 'unknown encoder' in lowered or 'error selecting an encoder' in lowered


def _filter_available_codecs(ffmpeg_bin: str, codecs: list[str]) -> list[str]:
    available = _available_encoders(ffmpeg_bin)
    if not available:
        return codecs
    filtered = [codec for codec in codecs if codec in available]
    return filtered or codecs


@lru_cache(maxsize=4)
def _available_encoders(ffmpeg_bin: str) -> set[str]:
    try:
        result = subprocess.run([ffmpeg_bin, '-encoders'], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, OSError):
        return set()

    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text or text.startswith('Encoders:'):
            continue
        parts = text.split()
        if len(parts) < 2:
            continue
        if not parts[0].startswith('V'):
            continue
        encoders.add(parts[1])
    return encoders


def _decoder_command(ffmpeg_bin: str, input_file: str) -> list[str]:
    return [
        ffmpeg_bin,
        '-v',
        'error',
        '-i',
        input_file,
        '-map',
        '0:v:0',
        '-vsync',
        '0',
        '-f',
        'rawvideo',
        '-pix_fmt',
        'bgr24',
        'pipe:1',
    ]


def _writer_command(
    ffmpeg_bin: str,
    output_file: str,
    input_file: str,
    fps: float,
    out_width: int,
    out_height: int,
    codec: str,
    pix_fmt: str,
    with_audio: bool,
) -> list[str]:
    command = [
        ffmpeg_bin,
        '-y',
        '-f',
        'rawvideo',
        '-pix_fmt',
        'bgr24',
        '-s',
        f'{out_width}x{out_height}',
        '-framerate',
        str(fps),
        '-i',
        'pipe:0',
    ]
    if with_audio:
        command.extend([
            '-i',
            input_file,
            '-map',
            '0:v:0',
            '-map',
            '1:a:0',
            '-c:a',
            'copy',
        ])
    command.extend([
        '-c:v',
        codec,
        '-r',
        str(fps),
        '-pix_fmt',
        pix_fmt,
        output_file,
    ])
    return command


def _read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def _drain_pipe(stream) -> str:
    if stream is None:
        return ''
    try:
        data = stream.read()
        if isinstance(data, bytes):
            return data.decode('utf-8', errors='replace').strip()
        return str(data).strip()
    except Exception:
        return ''


def _close_stdin(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.stdin is None:
        return
    try:
        process.stdin.close()
    except Exception:
        pass


def _terminate_process(process: Optional[subprocess.Popen]) -> None:
    if process is None:
        return
    if process.stdin is not None and not process.stdin.closed:
        try:
            process.stdin.close()
        except Exception:
            pass
    if process.stdout is not None:
        try:
            process.stdout.close()
        except Exception:
            pass
    if process.stderr is not None:
        try:
            process.stderr.close()
        except Exception:
            pass
    if process.poll() is None:
        try:
            process.kill()
        except Exception:
            pass
        try:
            process.wait(timeout=1)
        except Exception:
            pass


def _queue_put(queue_obj: Queue[Any], item: Any, stop_event: Optional[threading.Event]) -> None:
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        try:
            queue_obj.put(item, timeout=0.1)
            return
        except Full:
            continue


def _queue_get(queue_obj: Queue[Any], stop_event: threading.Event) -> Any:
    while True:
        if stop_event.is_set() and queue_obj.empty():
            return _QUEUE_EOF
        try:
            return queue_obj.get(timeout=0.1)
        except Empty:
            continue


def _should_fallback_to_extract(error: ServiceError) -> bool:
    return error.code == ErrorCode.FFMPEG_ERROR
