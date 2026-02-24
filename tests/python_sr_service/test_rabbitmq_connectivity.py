import os
import uuid

import pika


def _build_mq_url() -> str:
    env_url = os.getenv('MQ_URL', '').strip()
    if env_url:
        return env_url
    return 'amqp://admin:Admin%40123@localhost:5672/%2F'


def test_rabbitmq_connectivity():
    url = _build_mq_url()
    connection = pika.BlockingConnection(pika.URLParameters(url))
    channel = connection.channel()

    queue_name = f'sr.test.connectivity.{uuid.uuid4().hex}'
    channel.queue_declare(queue=queue_name, durable=False, auto_delete=True)

    payload = b'ping'
    channel.basic_publish(exchange='', routing_key=queue_name, body=payload)
    method, _, body = channel.basic_get(queue=queue_name, auto_ack=True)

    assert method is not None
    assert body == payload

    channel.queue_delete(queue=queue_name)
    connection.close()

