import pytest

from python_sr_service.config import Settings


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv('COS_SECRET_ID', 'id')
    monkeypatch.setenv('COS_SECRET_KEY', 'key')
    monkeypatch.setenv('COS_REGION', 'ap-guangzhou')
    monkeypatch.setenv('COS_BUCKET', 'bucket-12345')
    monkeypatch.setenv('COS_PREFIX', 'sr')
    settings = Settings.from_env(config_path='')
    assert settings.cos.secret_id == 'id'
    assert settings.cos.secret_key == 'key'
    assert settings.cos.region == 'ap-guangzhou'
    assert settings.cos.bucket == 'bucket-12345'
    assert settings.cos.prefix == 'sr'


def test_settings_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv('COS_SECRET_ID', raising=False)
    monkeypatch.delenv('COS_SECRET_KEY', raising=False)
    monkeypatch.delenv('COS_REGION', raising=False)
    monkeypatch.delenv('COS_BUCKET', raising=False)

    config_file = tmp_path / 'application.yml'
    config_file.write_text(
        'cos:\n'
        '  secret_id: file-id\n'
        '  secret_key: file-key\n'
        '  region: ap-shanghai\n'
        '  bucket: file-bucket-1\n'
        '  prefix: file-prefix\n',
        encoding='utf-8')

    settings = Settings.from_env(config_path=str(config_file))
    assert settings.cos.secret_id == 'file-id'
    assert settings.cos.secret_key == 'file-key'
    assert settings.cos.region == 'ap-shanghai'
    assert settings.cos.bucket == 'file-bucket-1'
    assert settings.cos.prefix == 'file-prefix'


def test_env_overrides_file(tmp_path, monkeypatch):
    config_file = tmp_path / 'application.yml'
    config_file.write_text(
        'cos:\n'
        '  secret_id: file-id\n'
        '  secret_key: file-key\n'
        '  region: ap-beijing\n'
        '  bucket: file-bucket-1\n',
        encoding='utf-8')

    monkeypatch.setenv('COS_SECRET_ID', 'env-id')
    monkeypatch.setenv('COS_REGION', 'ap-guangzhou')

    settings = Settings.from_env(config_path=str(config_file))
    assert settings.cos.secret_id == 'env-id'
    assert settings.cos.secret_key == 'file-key'
    assert settings.cos.region == 'ap-guangzhou'


def test_settings_missing_required_env(monkeypatch):
    monkeypatch.delenv('COS_SECRET_ID', raising=False)
    monkeypatch.delenv('COS_SECRET_KEY', raising=False)
    monkeypatch.delenv('COS_REGION', raising=False)
    monkeypatch.delenv('COS_BUCKET', raising=False)
    with pytest.raises(ValueError):
        Settings.from_env(config_path='')
