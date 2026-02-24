import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pika

from python_sr_service.config import Settings
from python_sr_service.idempotency.redis_store import RedisIdempotencyStore
from python_sr_service.persistence.mysql_event_repo import parse_mysql_dsn
from python_sr_service.storage.cos_client import TencentCOSClient
from python_sr_service.worker.consumer import RabbitMQConsumer
from python_sr_service.worker.publisher import RabbitMQResultPublisher


def ensure_mysql_table(settings: Settings) -> None:
    if not settings.mysql.dsn:
        return
    import pymysql

    conn_kwargs = parse_mysql_dsn(settings.mysql.dsn)
    conn_kwargs['connect_timeout'] = settings.mysql.connect_timeout_seconds
    conn_kwargs['read_timeout'] = settings.mysql.read_timeout_seconds
    conn_kwargs['write_timeout'] = settings.mysql.write_timeout_seconds
    conn_kwargs['autocommit'] = True

    ddl = (
        'CREATE TABLE IF NOT EXISTS sr_task_event ('
        'id BIGINT PRIMARY KEY AUTO_INCREMENT,'
        'task_id BIGINT NOT NULL,'
        'task_no VARCHAR(64) NOT NULL,'
        'event_type VARCHAR(32) NOT NULL,'
        'event_time DATETIME(6) NOT NULL,'
        'attempt INT NOT NULL DEFAULT 1,'
        'worker_id VARCHAR(128) NULL,'
        'payload_json JSON NULL,'
        'error_code VARCHAR(64) NULL,'
        'error_msg VARCHAR(1024) NULL,'
        'trace_id VARCHAR(128) NULL,'
        'created_at DATETIME(6) NOT NULL,'
        'KEY idx_task_id (task_id),'
        'KEY idx_trace_id (trace_id)'
        ') ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;'
    )
    with pymysql.connect(**conn_kwargs) as conn:
        with conn.cursor() as cursor:
            cursor.execute(ddl)


def fetch_event_count(settings: Settings, task_id: int) -> int:
    if not settings.mysql.dsn:
        return 0
    import pymysql

    conn_kwargs = parse_mysql_dsn(settings.mysql.dsn)
    conn_kwargs['autocommit'] = True
    with pymysql.connect(**conn_kwargs) as conn:
        with conn.cursor() as cursor:
            cursor.execute('SELECT COUNT(1) FROM sr_task_event WHERE task_id=%s', (task_id,))
            row = cursor.fetchone()
            return int(row[0]) if row else 0


def main() -> int:
    settings = Settings.from_env()
    ensure_mysql_table(settings)

    storage = TencentCOSClient(settings.cos)
    idempotency = RedisIdempotencyStore(settings.redis.url, settings.idempotency.ttl_seconds)
    publisher = RabbitMQResultPublisher(settings.rabbitmq)
    consumer = RabbitMQConsumer(
        settings=settings,
        idempotency_store=idempotency,
        publisher=publisher,
        storage=storage,
    )

    sample_input = Path('inputs/0014.jpg')
    if not sample_input.exists():
        raise RuntimeError(f'Sample input not found: {sample_input}')

    trace_id = uuid4().hex
    task_id = int(time.time())
    task_no = f'SR{task_id}'
    input_key = f'e2e/input/{trace_id}.jpg'

    storage.upload(str(sample_input), input_key)

    connection = pika.BlockingConnection(pika.URLParameters(settings.rabbitmq.url))
    channel = connection.channel()
    channel.exchange_declare(exchange=settings.rabbitmq.result_exchange, exchange_type='direct', durable=True)
    result_queue = f'sr.result.e2e.{uuid4().hex}'
    channel.queue_declare(queue=result_queue, durable=False, auto_delete=True)
    channel.queue_bind(
        queue=result_queue,
        exchange=settings.rabbitmq.result_exchange,
        routing_key=settings.rabbitmq.result_routing_key,
    )

    task_payload = {
        'schemaVersion': '1.0',
        'eventId': f'evt_task_{uuid4().hex}',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'taskId': task_id,
        'taskNo': task_no,
        'userId': 10001,
        'type': 'image',
        'inputFileKey': input_key,
        'scale': 4,
        'modelName': settings.inference.model_name,
        'modelVersion': 'v1.0.0',
        'attempt': 1,
        'traceId': trace_id,
    }

    channel.exchange_declare(exchange=settings.rabbitmq.task_exchange, exchange_type='direct', durable=True)
    channel.queue_declare(queue=settings.rabbitmq.task_queue, durable=True)
    channel.queue_bind(
        queue=settings.rabbitmq.task_queue,
        exchange=settings.rabbitmq.task_exchange,
        routing_key=settings.rabbitmq.task_routing_key,
    )
    channel.basic_publish(
        exchange=settings.rabbitmq.task_exchange,
        routing_key=settings.rabbitmq.task_routing_key,
        body=json.dumps(task_payload).encode('utf-8'),
        properties=pika.BasicProperties(content_type='application/json', delivery_mode=2),
    )

    consumed = consumer.consume_once(timeout_seconds=120)
    if not consumed:
        raise RuntimeError('Worker did not consume task within timeout')

    statuses = []
    output_key = None
    deadline = time.time() + 120
    while time.time() < deadline:
        method, _, body = channel.basic_get(queue=result_queue, auto_ack=True)
        if method is None:
            time.sleep(0.2)
            continue
        payload = json.loads(body.decode('utf-8'))
        if int(payload.get('taskId', 0)) != task_id:
            continue
        statuses.append(payload.get('status'))
        if payload.get('status') == 'SUCCEEDED':
            output_key = payload.get('outputFileKey')
            break
        if payload.get('status') == 'FAILED':
            raise RuntimeError(f"Task failed: errorCode={payload.get('errorCode')} errorMsg={payload.get('errorMsg')}")

    if output_key is None:
        raise RuntimeError(f'No SUCCEEDED result received. statuses={statuses}')

    if not storage.exists(output_key):
        raise RuntimeError(f'Output file not found in COS: {output_key}')

    event_count = fetch_event_count(settings, task_id)

    print('E2E_SUCCESS')
    print(f'task_id={task_id}')
    print(f'input_key={input_key}')
    print(f'output_key={output_key}')
    print(f'result_statuses={statuses}')
    print(f'event_count={event_count}')

    channel.queue_delete(queue=result_queue)
    connection.close()
    publisher.close()
    idempotency.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
