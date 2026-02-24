from typing import Optional


class RedisIdempotencyStore:
    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 259200,
        key_prefix: str = 'dedupe',
        client: Optional[object] = None,
    ):
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix
        self._client = client or self._build_client(redis_url)

    def is_processed(self, event_id: str) -> bool:
        key = self._key(event_id)
        return bool(self._client.exists(key))

    def mark_processed(self, event_id: str) -> bool:
        # 通过 NX 保证并发重复投递时“先写入者生效”。
        key = self._key(event_id)
        result = self._client.set(key, '1', nx=True, ex=self._ttl_seconds)
        return bool(result)

    def try_mark_processed(self, event_id: str) -> bool:
        # 兼容旧接口语义，保留历史调用方式。
        return self.mark_processed(event_id)

    def _key(self, event_id: str) -> str:
        return f'{self._key_prefix}:{event_id}'

    def clear(self, event_id: str) -> None:
        key = f'{self._key_prefix}:{event_id}'
        self._client.delete(key)

    def close(self) -> None:
        close_fn = getattr(self._client, 'close', None)
        if callable(close_fn):
            close_fn()

    @staticmethod
    def _build_client(redis_url: str):
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError('redis package is required. Please install: pip install redis') from exc
        return redis.Redis.from_url(redis_url, decode_responses=True)

