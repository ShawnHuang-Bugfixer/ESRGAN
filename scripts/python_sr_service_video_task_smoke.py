import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pika

from python_sr_service.config import Settings
from python_sr_service.idempotency.redis_store import RedisIdempotencyStore
from python_sr_service.runtime.logging import setup_logging
from python_sr_service.storage.cos_client import TencentCOSClient
from python_sr_service.worker.consumer import LOCKED_SERVICE_MODEL_NAME, RabbitMQConsumer
from python_sr_service.worker.publisher import RabbitMQResultPublisher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Publish a video SR task and run one-shot worker smoke test.')
    parser.add_argument('--input', default='inputs/video/onepiece_demo.mp4', help='Local video path to upload.')
    parser.add_argument('--scale', type=int, default=2, help='SR scale for video task.')
    parser.add_argument('--model', default=LOCKED_SERVICE_MODEL_NAME, help='Model name in task payload.')
    parser.add_argument('--timeout-seconds', type=int, default=300, help='Timeout for task consumption/result.')
    parser.add_argument('--publish-only', action='store_true', help='Only publish task message, do not consume.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings.from_env()
    setup_logging(settings.runtime.log_level)

    video_path = Path(args.input)
    if not video_path.exists():
        raise RuntimeError(f'Input video not found: {video_path}')

    storage = TencentCOSClient(settings.cos)
    idempotency = RedisIdempotencyStore(settings.redis.url, settings.idempotency.ttl_seconds)
    publisher = RabbitMQResultPublisher(settings.rabbitmq)
    consumer = RabbitMQConsumer(
        settings=settings,
        idempotency_store=idempotency,
        publisher=publisher,
        storage=storage,
    )
    consumer.prepare()

    trace_id = uuid4().hex
    task_id = int(time.time())
    task_no = f'SR{task_id}'
    input_key = f'e2e/video/{trace_id}{video_path.suffix.lower()}'

    storage.upload(str(video_path), input_key)

    connection = pika.BlockingConnection(pika.URLParameters(settings.rabbitmq.url))
    channel = connection.channel()

    channel.exchange_declare(exchange=settings.rabbitmq.result_exchange, exchange_type='direct', durable=True)
    result_queue = f'sr.result.video.e2e.{uuid4().hex}'
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
        'type': 'video',
        'inputFileKey': input_key,
        'scale': args.scale,
        'modelName': args.model,
        'modelVersion': 'v1.0.0',
        'attempt': 1,
        'traceId': trace_id,
        'videoOptions': {
            'keepAudio': True,
            'extractFrameFirst': False,
        },
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

    if args.publish_only:
        print('VIDEO_TASK_PAYLOAD')
        print(json.dumps(task_payload, ensure_ascii=False, indent=2))
        print('PUBLISHED_ONLY=true')
        channel.queue_delete(queue=result_queue)
        connection.close()
        publisher.close()
        idempotency.close()
        return 0

    consumed = consumer.consume_once(timeout_seconds=args.timeout_seconds)
    if not consumed:
        raise RuntimeError('Worker did not consume task within timeout')

    statuses = []
    final_payload = None
    deadline = time.time() + args.timeout_seconds
    while time.time() < deadline:
        method, _, body = channel.basic_get(queue=result_queue, auto_ack=True)
        if method is None:
            time.sleep(0.3)
            continue
        payload = json.loads(body.decode('utf-8'))
        if int(payload.get('taskId', 0)) != task_id:
            continue
        statuses.append(payload.get('status'))
        if payload.get('status') in ('SUCCEEDED', 'FAILED', 'CANCELLED'):
            final_payload = payload
            break

    if final_payload is None:
        raise RuntimeError(f'No final result received, statuses={statuses}')

    print('VIDEO_TASK_PAYLOAD')
    print(json.dumps(task_payload, ensure_ascii=False, indent=2))
    print('VIDEO_RESULT_PAYLOAD')
    print(json.dumps(final_payload, ensure_ascii=False, indent=2))
    print(f'ALL_STATUSES={statuses}')

    channel.queue_delete(queue=result_queue)
    connection.close()
    publisher.close()
    idempotency.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
