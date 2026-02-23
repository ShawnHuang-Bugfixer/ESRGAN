import uuid

from python_sr_service.config import Settings
from python_sr_service.storage.cos_client import TencentCOSClient


def test_cos_client_integration_all_methods(tmp_path):
    settings = Settings.from_env(config_path='python_sr_service/application.yml')
    client = TencentCOSClient(settings.cos)

    run_id = uuid.uuid4().hex[:12]
    object_key = f'integration/{run_id}/sample.bin'
    list_prefix = f'integration/{run_id}'
    payload = b'cos integration payload'

    src = tmp_path / 'source.bin'
    dst = tmp_path / 'downloaded.bin'
    src.write_bytes(payload)

    # upload + exists
    client.upload(str(src), object_key)
    assert client.exists(object_key) is True

    # list_objects
    keys = client.list_objects(list_prefix)
    expected_key = client._full_key(object_key)
    assert expected_key in keys

    # download
    client.download(object_key, str(dst))
    assert dst.read_bytes() == payload

    # delete + exists
    client.delete(object_key)
    assert client.exists(object_key) is False
