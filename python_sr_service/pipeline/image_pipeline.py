import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from python_sr_service.config import InferenceSettings
from python_sr_service.domain.errors import ErrorCode, ServiceError
from python_sr_service.runtime.logging import format_log_fields

logger = logging.getLogger(__name__)

_TILE_FALLBACKS = (1024, 768, 512, 384, 256, 192, 128, 64)
_STARTUP_PRELOAD_TILES = (0, 1024)


@dataclass(frozen=True)
class InferenceOutput:
    output_path: str
    cost_ms: int


class ImagePipeline:
    def __init__(self, settings: InferenceSettings):
        self._settings = settings
        self._upsamplers: Dict[str, RealESRGANer] = {}

    def prepare(self, model_name_override: str = '') -> None:
        model_name = self._resolve_model_name(model_name_override)
        preload_tiles = _startup_preload_tiles(self._settings.tile)
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True

        for tile in preload_tiles:
            upsampler = self._ensure_upsampler(model_name_override=model_name, tile_override=tile)
            if not _should_warmup(self._settings.device):
                continue

            warmup_image = np.zeros((16, 16, 3), dtype=np.uint8)
            start = time.time()
            try:
                with torch.inference_mode():
                    upsampler.enhance(warmup_image, outscale=4)
            except Exception:
                logger.exception('image_model_warmup_failed')
                continue

            logger.info(
                'image_model_warmup_done %s',
                format_log_fields(
                    {
                        'modelName': model_name,
                        'device': self._settings.device,
                        'tile': tile,
                        'costMs': int((time.time() - start) * 1000),
                    },
                ),
            )

        logger.info(
            'image_model_prepare_done %s',
            format_log_fields(
                {
                    'modelName': model_name,
                    'device': self._settings.device,
                    'preloadedTiles': preload_tiles,
                },
            ),
        )

    def run(self, input_file: str, output_file: str, scale: int, model_name_override: str = '') -> InferenceOutput:
        logger.info(
            'image_infer_start %s',
            format_log_fields(
                {
                    'phase': 'enhance',
                    'inputFile': input_file,
                    'outputFile': output_file,
                    'scale': scale,
                    'modelName': self._resolve_model_name(model_name_override),
                    'device': self._settings.device,
                },
            ),
        )

        decode_started = time.time()
        image = cv2.imread(input_file, cv2.IMREAD_UNCHANGED)
        decode_cost_ms = int((time.time() - decode_started) * 1000)
        if image is None:
            raise ServiceError(
                code=ErrorCode.INPUT_NOT_FOUND,
                message=f'Failed to read input image: {input_file}',
                retryable=False,
            )
        logger.info(
            'image_decode_done %s',
            format_log_fields(
                {
                    'phase': 'decode',
                    'inputFile': input_file,
                    'costMs': decode_cost_ms,
                },
            ),
        )

        output, cost_ms = self.enhance_array(image, scale, model_name_override=model_name_override)

        write_started = time.time()
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        if not cv2.imwrite(output_file, output):
            raise ServiceError(
                code=ErrorCode.INFER_RUNTIME_ERROR,
                message=f'Failed to write output image: {output_file}',
                retryable=False,
            )
        write_cost_ms = int((time.time() - write_started) * 1000)
        logger.info(
            'image_write_done %s',
            format_log_fields(
                {
                    'phase': 'encode_or_write',
                    'outputFile': output_file,
                    'costMs': write_cost_ms,
                },
            ),
        )

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

    def enhance_array(self, image, scale: int, model_name_override: str = ''):
        start = time.time()
        model_name = self._resolve_model_name(model_name_override)
        tile_candidates = _tile_candidates(self._settings.tile, image)
        last_oom: Optional[RuntimeError] = None

        for index, tile in enumerate(tile_candidates):
            upsampler = self._ensure_upsampler(model_name_override=model_name, tile_override=tile)
            try:
                with torch.inference_mode():
                    output, _ = upsampler.enhance(image, outscale=scale)
                cost_ms = int((time.time() - start) * 1000)
                if tile != self._settings.tile:
                    logger.warning(
                        'image_infer_tiled_fallback_succeeded %s',
                        format_log_fields(
                            {
                                'modelName': model_name,
                                'tile': tile,
                                'costMs': cost_ms,
                            },
                        ),
                    )
                return output, cost_ms
            except RuntimeError as exc:
                text = str(exc).lower()
                if 'out of memory' not in text:
                    raise ServiceError(
                        code=ErrorCode.INFER_RUNTIME_ERROR,
                        message='Runtime error during image inference',
                        retryable=True,
                        cause=exc,
                    ) from exc

                last_oom = exc
                next_tile = tile_candidates[index + 1] if index + 1 < len(tile_candidates) else None
                if next_tile is None:
                    break
                _clear_cuda_cache()
                logger.warning(
                    'image_infer_retry_tiled %s',
                    format_log_fields(
                        {
                            'modelName': model_name,
                            'device': self._settings.device,
                            'attemptedTile': tile,
                            'nextTile': next_tile,
                            'height': int(image.shape[0]),
                            'width': int(image.shape[1]),
                        },
                    ),
                )
            except Exception as exc:
                logger.exception('image_infer_unexpected_error')
                raise ServiceError(
                    code=ErrorCode.INFER_RUNTIME_ERROR,
                    message='Failed to run image inference',
                    retryable=True,
                    cause=exc,
                ) from exc

        raise ServiceError(
            code=ErrorCode.GPU_OOM,
            message='GPU out of memory during inference',
            retryable=False,
            cause=last_oom,
        ) from last_oom

    def _ensure_upsampler(self, model_name_override: str = '', tile_override: Optional[int] = None) -> RealESRGANer:
        model_name = self._resolve_model_name(model_name_override)
        tile = self._settings.tile if tile_override is None else int(tile_override)
        cache_key = _upsampler_cache_key(model_name, tile)
        if cache_key in self._upsamplers:
            return self._upsamplers[cache_key]

        model, net_scale = _build_model(model_name)
        model_path, dni_weight = _resolve_model_paths(self._settings, model_name_override=model_name)
        model_path = _materialize_model_paths(model_path)
        _validate_model_paths(model_path)

        gpu_id, use_half = _resolve_device(self._settings.device, self._settings.fp32)
        logger.info(
            'image_model_init %s',
            format_log_fields(
                {
                    'modelName': model_name,
                    'modelPath': model_path,
                    'netScale': net_scale,
                    'device': self._settings.device,
                    'gpuId': gpu_id,
                    'half': use_half,
                    'tile': tile,
                    'tilePad': self._settings.tile_pad,
                },
            ),
        )
        upsampler = RealESRGANer(
            scale=net_scale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=tile,
            tile_pad=self._settings.tile_pad,
            pre_pad=self._settings.pre_pad,
            half=use_half,
            gpu_id=gpu_id,
        )
        self._upsamplers[cache_key] = upsampler
        return upsampler

    def _resolve_model_name(self, model_name_override: str) -> str:
        preferred = (model_name_override or '').strip()
        if preferred:
            return preferred
        return self._settings.model_name.strip()


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


def _resolve_model_paths(
    settings: InferenceSettings,
    model_name_override: str = '',
) -> Tuple[Union[str, List[str]], Optional[List[float]]]:
    model_name = (model_name_override or settings.model_name).strip()
    if settings.model_weights.strip() and model_name == settings.model_name.strip():
        primary_path = settings.model_weights.strip()
    else:
        primary_path = os.path.join('weights', f'{model_name}.pth')

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


def _should_warmup(device: str) -> bool:
    lower = device.strip().lower()
    return lower.startswith('cuda') and torch.cuda.is_available()


def _tile_candidates(configured_tile: int, image) -> list[int]:
    max_side = max(int(image.shape[0]), int(image.shape[1]))
    candidates: list[int] = []
    first_tile = _initial_tile(configured_tile, image)
    candidates.append(first_tile)
    for tile in _TILE_FALLBACKS:
        if tile >= max_side:
            continue
        if first_tile > 0 and tile >= first_tile:
            continue
        if tile not in candidates:
            candidates.append(tile)
    return candidates


def _startup_preload_tiles(configured_tile: int) -> list[int]:
    base_tile = max(0, int(configured_tile))
    candidates: list[int] = [base_tile]
    for tile in _STARTUP_PRELOAD_TILES:
        normalized = max(0, int(tile))
        if normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _initial_tile(configured_tile: int, image) -> int:
    normalized_tile = max(0, int(configured_tile))
    if normalized_tile > 0:
        return normalized_tile

    height = int(image.shape[0])
    width = int(image.shape[1])
    max_side = max(height, width)
    pixels = height * width

    if max_side <= 1200 and pixels <= 1_500_000:
        return 0
    if max_side <= 2000 and pixels <= 3_500_000:
        return 1024
    if max_side <= 3000 and pixels <= 6_500_000:
        return 768
    if max_side <= 4200 and pixels <= 12_000_000:
        return 512
    return 384


def _upsampler_cache_key(model_name: str, tile: int) -> str:
    return f'{model_name}:tile={max(0, int(tile))}'


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
