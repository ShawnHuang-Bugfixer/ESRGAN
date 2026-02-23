from dataclasses import dataclass
import os
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = 'python_sr_service/application.yml'


@dataclass(frozen=True)
class COSSettings:
    secret_id: str
    secret_key: str
    region: str
    bucket: str
    scheme: str = 'https'
    token: str = ''
    prefix: str = ''
    endpoint: str = ''


@dataclass(frozen=True)
class Settings:
    cos: COSSettings

    @classmethod
    def from_env(cls, config_path: str = ''):
        resolved_path = config_path or os.getenv('SR_CONFIG_FILE', DEFAULT_CONFIG_PATH)
        config_data = _load_config_file(resolved_path)
        cos_config = config_data.get('cos', {})

        return cls(cos=COSSettings(
            secret_id=_get_value('COS_SECRET_ID', cos_config, 'secret_id', required=True),
            secret_key=_get_value('COS_SECRET_KEY', cos_config, 'secret_key', required=True),
            region=_get_value('COS_REGION', cos_config, 'region', required=True),
            bucket=_get_value('COS_BUCKET', cos_config, 'bucket', required=True),
            scheme=_get_value('COS_SCHEME', cos_config, 'scheme', default='https'),
            token=_get_value('COS_TOKEN', cos_config, 'token', default=''),
            prefix=_get_value('COS_PREFIX', cos_config, 'prefix', default=''),
            endpoint=_get_value('COS_ENDPOINT', cos_config, 'endpoint', default=''),
        ))


def _get_value(env_name: str, section: Dict[str, Any], key: str, required: bool = False, default: str = '') -> str:
    env_value = os.getenv(env_name, '').strip()
    if env_value:
        return env_value

    file_value = section.get(key)
    if isinstance(file_value, str) and file_value.strip():
        return file_value.strip()

    if file_value is not None and not isinstance(file_value, str):
        text_value = str(file_value).strip()
        if text_value:
            return text_value

    if required:
        raise ValueError(f'Missing required configuration: {env_name} (or cos.{key} in config file)')

    return default


def _load_config_file(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}

    with open(path, 'r', encoding='utf-8') as file_obj:
        data = yaml.safe_load(file_obj) or {}

    if not isinstance(data, dict):
        raise ValueError(f'Invalid config file format: {path}')

    return data
