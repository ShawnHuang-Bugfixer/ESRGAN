from datetime import datetime

import pytest

from python_sr_service.config import MySQLSettings
from python_sr_service.domain.errors import ErrorCode, ServiceError
from python_sr_service.persistence.mysql_event_repo import MySQLEventRepository, TaskEventRecord, parse_mysql_dsn


class FakeCursor:
    def __init__(self):
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params):
        self.executed.append((sql, params))


class FakeConnection:
    def __init__(self):
        self.cursor_obj = FakeCursor()
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True


def test_save_event_success():
    settings = MySQLSettings(dsn='mysql://root:pwd@localhost:3306/picture', autocommit=True)
    fake_connection = FakeConnection()
    repo = MySQLEventRepository(settings, connection_factory=lambda: fake_connection)

    repo.save_event(TaskEventRecord(
        task_id=1001,
        task_no='SR202602230001',
        event_type='RECEIVED',
        event_time=datetime(2026, 2, 23, 16, 0, 0),
        attempt=1,
        worker_id='worker-1',
        payload_json={'k': 'v'},
        trace_id='trace-001',
    ))

    assert len(fake_connection.cursor_obj.executed) == 1
    sql, params = fake_connection.cursor_obj.executed[0]
    assert 'INSERT INTO sr_task_event' in sql
    assert params[0] == 1001
    assert params[1] == 'SR202602230001'
    assert params[2] == 'RECEIVED'
    assert params[6] == '{"k": "v"}'
    assert params[9] == 'trace-001'


def test_save_event_failure_maps_to_service_error():
    settings = MySQLSettings(dsn='mysql://root:pwd@localhost:3306/picture')

    def _raise():
        raise RuntimeError('db is down')

    repo = MySQLEventRepository(settings, connection_factory=_raise)

    with pytest.raises(ServiceError) as exc_info:
        repo.save_event(TaskEventRecord(
            task_id=1,
            task_no='SR1',
            event_type='FAILED',
            event_time=datetime(2026, 2, 23, 16, 0, 0),
        ))

    assert exc_info.value.code == ErrorCode.MYSQL_WRITE_FAILED
    assert exc_info.value.retryable is True


def test_parse_mysql_dsn():
    dsn = 'mysql://root:pw%4023@127.0.0.1:3307/picture?charset=utf8mb4'
    parsed = parse_mysql_dsn(dsn)

    assert parsed['host'] == '127.0.0.1'
    assert parsed['port'] == 3307
    assert parsed['user'] == 'root'
    assert parsed['password'] == 'pw@23'
    assert parsed['database'] == 'picture'
    assert parsed['charset'] == 'utf8mb4'

