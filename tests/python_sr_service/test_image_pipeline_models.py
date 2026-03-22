import os

import numpy as np
import pytest

from python_sr_service.config import InferenceSettings
from python_sr_service.domain.errors import ErrorCode, ServiceError
from python_sr_service.pipeline.image_pipeline import (
    ImagePipeline,
    _build_model,
    _initial_tile,
    _materialize_model_paths,
    _resolve_model_paths,
    _startup_preload_tiles,
    _tile_candidates,
)


@pytest.mark.parametrize(
    ('model_name', 'expected_scale'),
    [
        ('RealESRGAN_x4plus', 4),
        ('RealESRNet_x4plus', 4),
        ('RealESRGAN_x4plus_anime_6B', 4),
        ('RealESRGAN_x2plus', 2),
        ('realesr-animevideov3', 4),
        ('realesr-general-x4v3', 4),
    ],
)
def test_build_model_supported(model_name, expected_scale):
    _, scale = _build_model(model_name)
    assert scale == expected_scale


def test_build_model_unsupported():
    with pytest.raises(ServiceError) as exc_info:
        _build_model('not-exist-model')
    assert exc_info.value.code == ErrorCode.MODEL_NOT_FOUND


def test_resolve_model_paths_default_single_path():
    settings = InferenceSettings(
        model_name='RealESRGAN_x4plus',
        model_weights='',
        denoise_strength=1.0,
    )
    model_path, dni_weight = _resolve_model_paths(settings)
    assert model_path == os.path.join('weights', 'RealESRGAN_x4plus.pth')
    assert dni_weight is None


def test_resolve_model_paths_general_with_dni():
    settings = InferenceSettings(
        model_name='realesr-general-x4v3',
        model_weights='weights/realesr-general-x4v3.pth',
        denoise_strength=0.3,
    )
    model_path, dni_weight = _resolve_model_paths(settings)
    assert model_path == [
        'weights/realesr-general-x4v3.pth',
        'weights/realesr-general-wdn-x4v3.pth',
    ]
    assert dni_weight == [0.3, 0.7]


def test_resolve_model_paths_general_denoise_out_of_range():
    settings = InferenceSettings(
        model_name='realesr-general-x4v3',
        model_weights='weights/realesr-general-x4v3.pth',
        denoise_strength=1.5,
    )
    with pytest.raises(ServiceError) as exc_info:
        _resolve_model_paths(settings)
    assert exc_info.value.code == ErrorCode.MODEL_NOT_FOUND


def test_materialize_model_paths_resolves_project_relative_weights():
    model_path = _materialize_model_paths(os.path.join('weights', 'RealESRGAN_x4plus.pth'))
    assert os.path.isabs(model_path)
    assert os.path.exists(model_path)


def test_initial_tile_keeps_whole_image_for_small_inputs():
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    assert _initial_tile(0, image) == 0


def test_initial_tile_uses_1024_for_medium_large_inputs():
    image = np.zeros((1820, 1280, 3), dtype=np.uint8)
    assert _initial_tile(0, image) == 1024


def test_initial_tile_uses_768_for_larger_inputs():
    image = np.zeros((2048, 2048, 3), dtype=np.uint8)
    assert _initial_tile(0, image) == 768


def test_initial_tile_respects_explicit_configured_tile():
    image = np.zeros((5000, 4000, 3), dtype=np.uint8)
    assert _initial_tile(512, image) == 512


def test_tile_candidates_add_fallbacks_for_large_images():
    image = np.zeros((3000, 2000, 3), dtype=np.uint8)

    assert _tile_candidates(0, image) == [768, 512, 384, 256, 192, 128, 64]


def test_startup_preload_tiles_includes_default_and_first_fallback():
    assert _startup_preload_tiles(0) == [0, 1024]
    assert _startup_preload_tiles(512) == [512, 0, 1024]


def test_enhance_array_retries_with_smaller_tile_on_gpu_oom(monkeypatch):
    pipeline = ImagePipeline(InferenceSettings(tile=0, device='cuda:0'))
    image = np.zeros((2048, 2048, 3), dtype=np.uint8)
    seen_tiles = []

    class FakeUpsampler:
        def __init__(self, tile: int):
            self.tile = tile

        def enhance(self, _image, outscale=4):
            seen_tiles.append(self.tile)
            if self.tile == 768:
                raise RuntimeError('CUDA out of memory')
            return np.zeros((32, 32, 3), dtype=np.uint8), None

    monkeypatch.setattr(pipeline, '_ensure_upsampler', lambda model_name_override='', tile_override=None: FakeUpsampler(tile_override))
    monkeypatch.setattr('python_sr_service.pipeline.image_pipeline._clear_cuda_cache', lambda: None)

    output, _ = pipeline.enhance_array(image, scale=4, model_name_override='RealESRGAN_x4plus')

    assert output.shape == (32, 32, 3)
    assert seen_tiles[:2] == [768, 512]


def test_enhance_array_raises_gpu_oom_when_all_tiles_fail(monkeypatch):
    pipeline = ImagePipeline(InferenceSettings(tile=256, device='cuda:0'))
    image = np.zeros((4096, 4096, 3), dtype=np.uint8)

    class FakeUpsampler:
        def enhance(self, _image, outscale=4):
            raise RuntimeError('CUDA out of memory')

    monkeypatch.setattr(pipeline, '_ensure_upsampler', lambda model_name_override='', tile_override=None: FakeUpsampler())
    monkeypatch.setattr('python_sr_service.pipeline.image_pipeline._clear_cuda_cache', lambda: None)

    with pytest.raises(ServiceError) as exc_info:
        pipeline.enhance_array(image, scale=4, model_name_override='RealESRGAN_x4plus')

    assert exc_info.value.code == ErrorCode.GPU_OOM
