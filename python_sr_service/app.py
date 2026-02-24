import logging

from python_sr_service.config import Settings
from python_sr_service.idempotency.redis_store import RedisIdempotencyStore
from python_sr_service.worker.consumer import RabbitMQConsumer
from python_sr_service.runtime.logging import setup_logging
from python_sr_service.worker.publisher import RabbitMQResultPublisher


def main() -> None:
    # 单进程入口：按统一配置初始化运行所需依赖。
    settings = Settings.from_env()
    setup_logging(settings.runtime.log_level)
    logger = logging.getLogger(__name__)
    logger.info('python_sr_service_start')
    idempotency_store = RedisIdempotencyStore(
        redis_url=settings.redis.url,
        ttl_seconds=settings.idempotency.ttl_seconds,
    )
    publisher = RabbitMQResultPublisher(settings.rabbitmq)
    consumer = RabbitMQConsumer(
        settings=settings,
        idempotency_store=idempotency_store,
        publisher=publisher,
    )
    consumer.start()
    logger.info('python_sr_service_stop')


if __name__ == '__main__':
    main()


