from pathlib import Path

import python_sr_service.config as config_module
import pytest

from python_sr_service.config import Settings


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv('COS_SECRET_ID', 'id')
    monkeypatch.setenv('COS_SECRET_KEY', 'key')
    monkeypatch.setenv('COS_REGION', 'ap-guangzhou')
    monkeypatch.setenv('COS_BUCKET', 'bucket-12345')
    monkeypatch.setenv('COS_PREFIX', 'sr')
    monkeypatch.setenv('COS_TIMEOUT_SECONDS', '240')
    monkeypatch.setenv('COS_MULTIPART_THRESHOLD_MB', '16')
    monkeypatch.setenv('COS_UPLOAD_PART_MB', '8')
    monkeypatch.setenv('COS_UPLOAD_MAX_THREAD', '4')
    monkeypatch.setenv('MYSQL_DSN', 'mysql://root:pwd@127.0.0.1:3306/picture')
    monkeypatch.setenv('MYSQL_AUTOCOMMIT', 'false')
    monkeypatch.setenv('MQ_URL', 'amqp://guest:guest@localhost:5672/%2F')
    monkeypatch.setenv('REDIS_URL', 'redis://localhost:6379/0')
    monkeypatch.setenv('MODEL_DENOISE_STRENGTH', '0.6')
    monkeypatch.setenv('VIDEO_ENABLED', 'true')
    monkeypatch.setenv('FFMPEG_BIN', 'ffmpeg')
    monkeypatch.setenv('MAX_VIDEO_FRAMES', '6000')
    monkeypatch.setenv('VIDEO_CODEC_FALLBACKS', 'mpeg4,libxvid')

    settings = Settings.from_env(config_path='')
    assert settings.cos.secret_id == 'id'
    assert settings.cos.secret_key == 'key'
    assert settings.cos.region == 'ap-guangzhou'
    assert settings.cos.bucket == 'bucket-12345'
    assert settings.cos.prefix == 'sr'
    assert settings.cos.timeout_seconds == 240
    assert settings.cos.multipart_threshold_mb == 16
    assert settings.cos.upload_part_mb == 8
    assert settings.cos.upload_max_thread == 4
    assert settings.mysql.dsn == 'mysql://root:pwd@127.0.0.1:3306/picture'
    assert settings.mysql.autocommit is False
    assert settings.rabbitmq.url == 'amqp://guest:guest@localhost:5672/%2F'
    assert settings.redis.url == 'redis://localhost:6379/0'
    assert settings.inference.model_weights == ''
    assert settings.inference.denoise_strength == 0.6
    assert settings.inference.video_enabled is True
    assert Path(settings.inference.ffmpeg_bin).name.lower() in {'ffmpeg', 'ffmpeg.exe'}
    assert settings.inference.max_video_frames == 6000
    assert settings.inference.video_codec_fallbacks == ('mpeg4', 'libxvid')


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
        '  timeout_seconds: 300\n'
        '  multipart_threshold_mb: 12\n'
        '  upload_part_mb: 6\n'
        '  upload_max_thread: 2\n'
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
        '  denoise_strength: 0.8\n'
        '  video_enabled: true\n'
        '  max_video_seconds: 300\n'
        '  video_codec_fallbacks: mpeg4,libxvid\n',
        encoding='utf-8')

    settings = Settings.from_env(config_path=str(config_file))
    assert settings.cos.secret_id == 'file-id'
    assert settings.cos.secret_key == 'file-key'
    assert settings.cos.region == 'ap-shanghai'
    assert settings.cos.bucket == 'file-bucket-1'
    assert settings.cos.prefix == 'file-prefix'
    assert settings.cos.timeout_seconds == 300
    assert settings.cos.multipart_threshold_mb == 12
    assert settings.cos.upload_part_mb == 6
    assert settings.cos.upload_max_thread == 2
    assert settings.mysql.dsn == 'mysql://root:pwd@localhost:3306/picture'
    assert settings.mysql.autocommit is False
    assert settings.rabbitmq.url == 'amqp://guest:guest@localhost:5672/%2F'
    assert settings.redis.url == 'redis://localhost:6379/0'
    assert settings.idempotency.ttl_seconds == 123
    assert settings.inference.model_name == 'realesr-general-x4v3'
    assert settings.inference.denoise_strength == 0.8
    assert settings.inference.video_enabled is True
    assert settings.inference.max_video_seconds == 300
    assert settings.inference.video_codec_fallbacks == ('mpeg4', 'libxvid')


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


def test_settings_resolve_ffmpeg_bins_from_conda_env(tmp_path, monkeypatch):
    env_root = tmp_path / 'env'
    library_bin = env_root / 'Library' / 'bin'
    library_bin.mkdir(parents=True)
    python_exe = env_root / 'python.exe'
    python_exe.write_text('', encoding='utf-8')
    ffmpeg = library_bin / 'ffmpeg.exe'
    ffprobe = library_bin / 'ffprobe.exe'
    ffmpeg.write_text('', encoding='utf-8')
    ffprobe.write_text('', encoding='utf-8')

    monkeypatch.setenv('COS_SECRET_ID', 'id')
    monkeypatch.setenv('COS_SECRET_KEY', 'key')
    monkeypatch.setenv('COS_REGION', 'ap-guangzhou')
    monkeypatch.setenv('COS_BUCKET', 'bucket-12345')
    monkeypatch.setenv('MQ_URL', 'amqp://guest:guest@localhost:5672/%2F')
    monkeypatch.setenv('REDIS_URL', 'redis://localhost:6379/0')
    monkeypatch.setenv('FFMPEG_BIN', 'ffmpeg')
    monkeypatch.setenv('FFPROBE_BIN', 'ffprobe')
    monkeypatch.setattr(config_module.shutil, 'which', lambda _: None)
    monkeypatch.setattr(config_module.sys, 'executable', str(python_exe))

    settings = Settings.from_env(config_path='')

    assert settings.inference.ffmpeg_bin == str(ffmpeg.resolve())
    assert settings.inference.ffprobe_bin == str(ffprobe.resolve())
