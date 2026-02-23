import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs, unquote, urlparse

from python_sr_service.config import MySQLSettings
from python_sr_service.domain.errors import ErrorCode, ServiceError


@dataclass(frozen=True)
class TaskEventRecord:
    task_id: int
    task_no: str
    event_type: str
    event_time: datetime
    attempt: int = 1
    worker_id: str = ''
    payload_json: Optional[Dict[str, Any]] = None
    error_code: str = ''
    error_msg: str = ''
    trace_id: str = ''
    created_at: Optional[datetime] = None


class MySQLEventRepository:
    def __init__(
        self,
        settings: MySQLSettings,
        connection_factory: Optional[Callable[[], Any]] = None,
    ):
        self._settings = settings
        self._connection_factory = connection_factory or _build_connection_factory(settings)

    def save_event(self, event: TaskEventRecord) -> None:
        sql = (
            'INSERT INTO sr_task_event '
            '(task_id, task_no, event_type, event_time, attempt, worker_id, payload_json, '
            'error_code, error_msg, trace_id, created_at) '
            'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
        )
        created_at = event.created_at or datetime.utcnow()
        payload_json = json.dumps(event.payload_json, ensure_ascii=False) if event.payload_json is not None else None
        params = (
            event.task_id,
            event.task_no,
            event.event_type,
            event.event_time,
            event.attempt,
            _nullable(event.worker_id),
            payload_json,
            _nullable(event.error_code),
            _nullable(event.error_msg),
            _nullable(event.trace_id),
            created_at,
        )

        try:
            with self._connection_factory() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                if not self._settings.autocommit:
                    conn.commit()
        except Exception as exc:
            raise ServiceError(
                code=ErrorCode.MYSQL_WRITE_FAILED,
                message=f'Failed to write sr_task_event for task_id={event.task_id}, event_type={event.event_type}',
                retryable=True,
                cause=exc,
            ) from exc


def parse_mysql_dsn(dsn: str) -> Dict[str, Any]:
    if not dsn:
        raise ValueError('MYSQL_DSN is empty')

    parsed = urlparse(dsn)
    if parsed.scheme not in ('mysql', 'mysql+pymysql'):
        raise ValueError(f'Unsupported MYSQL_DSN scheme: {parsed.scheme}')

    database = parsed.path.lstrip('/')
    if not database:
        raise ValueError('MYSQL_DSN must include database name, e.g. mysql://user:pwd@host:3306/picture')

    query_values = parse_qs(parsed.query)
    charset = _first_or_default(query_values.get('charset'), 'utf8mb4')

    return {
        'host': parsed.hostname or '127.0.0.1',
        'port': parsed.port or 3306,
        'user': unquote(parsed.username or ''),
        'password': unquote(parsed.password or ''),
        'database': database,
        'charset': charset,
    }


def _build_connection_factory(settings: MySQLSettings) -> Callable[[], Any]:
    connect_kwargs = parse_mysql_dsn(settings.dsn)
    connect_kwargs['connect_timeout'] = settings.connect_timeout_seconds
    connect_kwargs['read_timeout'] = settings.read_timeout_seconds
    connect_kwargs['write_timeout'] = settings.write_timeout_seconds
    connect_kwargs['autocommit'] = settings.autocommit

    def _create_connection():
        try:
            import pymysql
        except ImportError as exc:
            raise RuntimeError('PyMySQL is required for MySQL integration. Please install: pip install pymysql') from exc
        return pymysql.connect(**connect_kwargs)

    return _create_connection


def _first_or_default(values, default: str) -> str:
    if not values:
        return default
    value = values[0].strip()
    return value or default


def _nullable(value: str) -> Optional[str]:
    normalized = (value or '').strip()
    return normalized or None

