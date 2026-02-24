import pytest

from python_sr_service.config import Settings


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv('COS_SECRET_ID', 'id')
    monkeypatch.setenv('COS_SECRET_KEY', 'key')
    monkeypatch.setenv('COS_REGION', 'ap-guangzhou')
    monkeypatch.setenv('COS_BUCKET', 'bucket-12345')
    monkeypatch.setenv('COS_PREFIX', 'sr')
    monkeypatch.setenv('MYSQL_DSN', 'mysql://root:pwd@127.0.0.1:3306/picture')
    monkeypatch.setenv('MYSQL_AUTOCOMMIT', 'false')
    monkeypatch.setenv('MQ_URL', 'amqp://guest:guest@localhost:5672/%2F')
    monkeypatch.setenv('REDIS_URL', 'redis://localhost:6379/0')
    monkeypatch.setenv('MODEL_DENOISE_STRENGTH', '0.6')
    settings = Settings.from_env(config_path='')
    assert settings.cos.secret_id == 'id'
    assert settings.cos.secret_key == 'key'
    assert settings.cos.region == 'ap-guangzhou'
    assert settings.cos.bucket == 'bucket-12345'
    assert settings.cos.prefix == 'sr'
    assert settings.mysql.dsn == 'mysql://root:pwd@127.0.0.1:3306/picture'
    assert settings.mysql.autocommit is False
    assert settings.rabbitmq.url == 'amqp://guest:guest@localhost:5672/%2F'
    assert settings.redis.url == 'redis://localhost:6379/0'
    assert settings.inference.model_weights == ''
    assert settings.inference.denoise_strength == 0.6


def test_settings_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv('COS_SECRET_ID', raising=False)
    monkeypatch.delenv('COS_SECRET_KEY', raising=False)
    monkeypatch.delenv('COS_REGION', raising=False)
    monkeypatch.delenv('COS_BUCKET', raising=False)
    monkeypatch.delenv('MYSQL_DSN', raising=False)
    monkeypatch.delenv('MQ_URL', raising=False)
    monkeypatch.delenv('REDIS_URL', raising=False)

    config_file = tmp_path / 'application.yml'
    config_file.write_text(
        'cos:\n'
        '  secret_id: file-id\n'
        '  secret_key: file-key\n'
        '  region: ap-shanghai\n'
        '  bucket: file-bucket-1\n'
        '  prefix: file-prefix\n'
        'mysql:\n'
        '  dsn: mysql://root:pwd@localhost:3306/picture\n'
        '  autocommit: false\n'
        'rabbitmq:\n'
        '  url: amqp://guest:guest@localhost:5672/%2F\n'
        'redis:\n'
        '  url: redis://localhost:6379/0\n'
        'idempotency:\n'
        '  ttl_seconds: 123\n'
        'inference:\n'
        '  model_name: realesr-general-x4v3\n'
        '  denoise_strength: 0.8\n',
        encoding='utf-8')

    settings = Settings.from_env(config_path=str(config_file))
    assert settings.cos.secret_id == 'file-id'
    assert settings.cos.secret_key == 'file-key'
    assert settings.cos.region == 'ap-shanghai'
    assert settings.cos.bucket == 'file-bucket-1'
    assert settings.cos.prefix == 'file-prefix'
    assert settings.mysql.dsn == 'mysql://root:pwd@localhost:3306/picture'
    assert settings.mysql.autocommit is False
    assert settings.rabbitmq.url == 'amqp://guest:guest@localhost:5672/%2F'
    assert settings.redis.url == 'redis://localhost:6379/0'
    assert settings.idempotency.ttl_seconds == 123
    assert settings.inference.model_name == 'realesr-general-x4v3'
    assert settings.inference.denoise_strength == 0.8


def test_env_overrides_file(tmp_path, monkeypatch):
    config_file = tmp_path / 'application.yml'
    config_file.write_text(
        'cos:\n'
        '  secret_id: file-id\n'
        '  secret_key: file-key\n'
        '  region: ap-beijing\n'
        '  bucket: file-bucket-1\n'
        'mysql:\n'
        '  dsn: mysql://file-user:file-pass@localhost:3306/picture\n'
        'rabbitmq:\n'
        '  url: amqp://guest:guest@localhost:5672/%2F\n'
        'redis:\n'
        '  url: redis://localhost:6379/0\n',
        encoding='utf-8')

    monkeypatch.setenv('COS_SECRET_ID', 'env-id')
    monkeypatch.setenv('COS_REGION', 'ap-guangzhou')
    monkeypatch.setenv('MYSQL_DSN', 'mysql://env-user:env-pass@127.0.0.1:3306/picture')

    settings = Settings.from_env(config_path=str(config_file))
    assert settings.cos.secret_id == 'env-id'
    assert settings.cos.secret_key == 'file-key'
    assert settings.cos.region == 'ap-guangzhou'
    assert settings.mysql.dsn == 'mysql://env-user:env-pass@127.0.0.1:3306/picture'
    assert settings.rabbitmq.url == 'amqp://guest:guest@localhost:5672/%2F'
    assert settings.redis.url == 'redis://localhost:6379/0'


def test_settings_missing_required_env(monkeypatch):
    monkeypatch.delenv('COS_SECRET_ID', raising=False)
    monkeypatch.delenv('COS_SECRET_KEY', raising=False)
    monkeypatch.delenv('COS_REGION', raising=False)
    monkeypatch.delenv('COS_BUCKET', raising=False)
    monkeypatch.delenv('MQ_URL', raising=False)
    monkeypatch.delenv('REDIS_URL', raising=False)
    with pytest.raises(ValueError):
        Settings.from_env(config_path='__missing__.yml')
