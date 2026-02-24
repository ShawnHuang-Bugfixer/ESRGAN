import os

import pytest

from python_sr_service.config import InferenceSettings
from python_sr_service.domain.errors import ErrorCode, ServiceError
from python_sr_service.pipeline.image_pipeline import _build_model, _materialize_model_paths, _resolve_model_paths


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
