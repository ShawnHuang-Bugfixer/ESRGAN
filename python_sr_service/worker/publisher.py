import json
import logging
from typing import Any, Dict, Optional

import pika
from pika import exceptions as pika_exceptions

from python_sr_service.config import RabbitMQSettings
from python_sr_service.runtime.logging import format_log_fields

logger = logging.getLogger(__name__)


class RabbitMQResultPublisher:
    def __init__(
        self,
        settings: RabbitMQSettings,
        connection: Optional[pika.BlockingConnection] = None,
    ):
        self._settings = settings
        self._connection = connection
        self._channel = None
        self._ensure_channel()
        logger.info(
            'result_publisher_ready %s',
            format_log_fields(
                {
                    'exchange': self._settings.result_exchange,
                    'routingKey': self._settings.result_routing_key,
                },
            ),
        )

    def _ensure_channel(self) -> None:
        if (
            self._channel is not None
            and getattr(self._channel, 'is_open', False)
            and self._connection is not None
            and self._connection.is_open
        ):
            return

        self._reset_connection()
        self._connection = pika.BlockingConnection(pika.URLParameters(self._settings.url))
        self._channel = self._connection.channel()
        self._channel.exchange_declare(
            exchange=self._settings.result_exchange,
            exchange_type='direct',
            durable=True,
        )

    def _reset_connection(self) -> None:
        if self._channel is not None:
            try:
                if self._channel.is_open:
                    self._channel.close()
            except Exception:
                pass
            finally:
                self._channel = None

        if self._connection is not None:
            try:
                if self._connection.is_open:
                    self._connection.close()
            except Exception:
                pass
            finally:
                self._connection = None

    def publish_result(self, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')

        last_error: Optional[Exception] = None
        for attempt in range(2):
            try:
                self._ensure_channel()
                self._channel.basic_publish(
                    exchange=self._settings.result_exchange,
                    routing_key=self._settings.result_routing_key,
                    body=body,
                    properties=pika.BasicProperties(
                        content_type='application/json',
                        delivery_mode=2,
                    ),
                )
                logger.info(
                    'result_published %s',
                    format_log_fields(
                        {
                            'traceId': payload.get('traceId', ''),
                            'taskId': payload.get('taskId', 0),
                            'status': payload.get('status', ''),
                            'progress': payload.get('progress', 0),
                            'costMs': payload.get('costMs', 0),
                            'errorCode': payload.get('errorCode'),
                        },
                    ),
                )
                return
            except (pika_exceptions.AMQPError, OSError) as exc:
                last_error = exc
                self._reset_connection()
                if attempt == 0:
                    logger.warning(
                        'result_publish_retry %s',
                        format_log_fields(
                            {
                                'traceId': payload.get('traceId', ''),
                                'taskId': payload.get('taskId', 0),
                                'error': str(exc),
                            },
                        ),
                    )
                    continue
                raise

        if last_error is not None:
            raise last_error

    def close(self) -> None:
        self._reset_connection()
        logger.info('result_publisher_closed')
