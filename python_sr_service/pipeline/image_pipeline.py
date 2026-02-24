import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from python_sr_service.config import InferenceSettings
from python_sr_service.domain.errors import ErrorCode, ServiceError
from python_sr_service.runtime.logging import format_log_fields

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceOutput:
    output_path: str
    cost_ms: int


class ImagePipeline:
    def __init__(self, settings: InferenceSettings):
        self._settings = settings
        self._upsampler = None

    def run(self, input_file: str, output_file: str, scale: int) -> InferenceOutput:
        # Lazy init avoids loading model weights before the first real task.
        self._ensure_upsampler()

        logger.info(
            'image_infer_start %s',
            format_log_fields(
                {
                    'phase': 'enhance',
                    'inputFile': input_file,
                    'outputFile': output_file,
                    'scale': scale,
                    'modelName': self._settings.model_name,
                    'device': self._settings.device,
                },
            ),
        )

        image = cv2.imread(input_file, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ServiceError(
                code=ErrorCode.INPUT_NOT_FOUND,
                message=f'Failed to read input image: {input_file}',
                retryable=False,
            )

        start = time.time()
        try:
            output, _ = self._upsampler.enhance(image, outscale=scale)
        except RuntimeError as exc:
            # Separate OOM from generic runtime errors for better handling upstream.
            text = str(exc).lower()
            if 'out of memory' in text:
                raise ServiceError(
                    code=ErrorCode.GPU_OOM,
                    message='GPU out of memory during inference',
                    retryable=False,
                    cause=exc,
                ) from exc
            raise ServiceError(
                code=ErrorCode.INFER_RUNTIME_ERROR,
                message='Runtime error during image inference',
                retryable=True,
                cause=exc,
            ) from exc
        except Exception as exc:
            raise ServiceError(
                code=ErrorCode.INFER_RUNTIME_ERROR,
                message='Failed to run image inference',
                retryable=True,
                cause=exc,
            ) from exc

        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        if not cv2.imwrite(output_file, output):
            raise ServiceError(
                code=ErrorCode.INFER_RUNTIME_ERROR,
                message=f'Failed to write output image: {output_file}',
                retryable=False,
            )

        cost_ms = int((time.time() - start) * 1000)
        logger.info(
            'image_infer_done %s',
            format_log_fields(
                {
                    'phase': 'enhance',
                    'status': 'DONE',
                    'outputFile': output_file,
                    'costMs': cost_ms,
                },
            ),
        )

        return InferenceOutput(
            output_path=output_file,
            cost_ms=cost_ms,
        )

    def _ensure_upsampler(self) -> None:
        if self._upsampler is not None:
            return

        model, net_scale = _build_model(self._settings.model_name)
        model_path, dni_weight = _resolve_model_paths(self._settings)
        model_path = _materialize_model_paths(model_path)
        _validate_model_paths(model_path)

        gpu_id, use_half = _resolve_device(self._settings.device, self._settings.fp32)
        logger.info(
            'image_model_init %s',
            format_log_fields(
                {
                    'modelName': self._settings.model_name,
                    'modelPath': model_path,
                    'netScale': net_scale,
                    'device': self._settings.device,
                    'gpuId': gpu_id,
                    'half': use_half,
                    'tile': self._settings.tile,
                    'tilePad': self._settings.tile_pad,
                },
            ),
        )
        # Reuse upsampler in-process to avoid repeated model loading overhead.
        self._upsampler = RealESRGANer(
            scale=net_scale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=self._settings.tile,
            tile_pad=self._settings.tile_pad,
            pre_pad=self._settings.pre_pad,
            half=use_half,
            gpu_id=gpu_id,
        )


def _build_model(model_name: str) -> Tuple[object, int]:
    normalized = model_name.strip()
    if normalized == 'RealESRGAN_x4plus':
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4
    if normalized == 'RealESRNet_x4plus':
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4
    if normalized == 'RealESRGAN_x4plus_anime_6B':
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4), 4
    if normalized == 'RealESRGAN_x2plus':
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2), 2
    if normalized == 'realesr-animevideov3':
        return (
            SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=16,
                upscale=4,
                act_type='prelu',
            ),
            4,
        )
    if normalized == 'realesr-general-x4v3':
        return (
            SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=4,
                act_type='prelu',
            ),
            4,
        )

    raise ServiceError(
        code=ErrorCode.MODEL_NOT_FOUND,
        message=f'Unsupported model name: {model_name}',
        retryable=False,
    )


def _resolve_model_paths(settings: InferenceSettings) -> Tuple[Union[str, List[str]], Optional[List[float]]]:
    model_name = settings.model_name.strip()
    primary_path = settings.model_weights.strip() or os.path.join('weights', f'{model_name}.pth')

    if model_name != 'realesr-general-x4v3':
        return primary_path, None

    denoise_strength = settings.denoise_strength
    if denoise_strength < 0 or denoise_strength > 1:
        raise ServiceError(
            code=ErrorCode.MODEL_NOT_FOUND,
            message=f'Invalid denoise_strength: {denoise_strength}. Expected value in [0, 1].',
            retryable=False,
        )

    if denoise_strength == 1:
        return primary_path, None

    wdn_path = _build_wdn_model_path(primary_path)
    return [primary_path, wdn_path], [denoise_strength, 1 - denoise_strength]


def _materialize_model_paths(model_path: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(model_path, str):
        return _resolve_model_path(model_path)
    return [_resolve_model_path(path) for path in model_path]


def _resolve_model_path(path: str) -> str:
    if os.path.isabs(path):
        return path

    cwd_candidate = os.path.abspath(path)
    if os.path.exists(cwd_candidate):
        return cwd_candidate

    project_candidate = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', path),
    )
    if os.path.exists(project_candidate):
        return project_candidate

    return project_candidate


def _build_wdn_model_path(primary_path: str) -> str:
    filename = os.path.basename(primary_path)
    if 'realesr-general-x4v3' in filename:
        return primary_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
    directory = os.path.dirname(primary_path)
    return os.path.join(directory, 'realesr-general-wdn-x4v3.pth')


def _validate_model_paths(model_path: Union[str, List[str]]) -> None:
    paths = [model_path] if isinstance(model_path, str) else model_path
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise ServiceError(
            code=ErrorCode.MODEL_NOT_FOUND,
            message=f'Model weights not found: {", ".join(missing)}',
            retryable=False,
        )


def _resolve_device(device: str, fp32: bool):
    lower = device.strip().lower()
    if lower == 'cpu':
        return None, False
    if lower.startswith('cuda') and torch.cuda.is_available():
        gpu_id = None
        if ':' in lower:
            try:
                gpu_id = int(lower.split(':', 1)[1])
            except ValueError:
                gpu_id = None
        return gpu_id, not fp32
    return None, False
