from python_sr_service.idempotency.redis_store import RedisIdempotencyStore


class FakeRedisClient:
    def __init__(self):
        self._keys = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self._keys:
            return False
        self._keys[key] = (value, ex)
        return True

    def exists(self, key):
        return 1 if key in self._keys else 0

    def delete(self, key):
        self._keys.pop(key, None)


def test_try_mark_processed_once():
    store = RedisIdempotencyStore(
        redis_url='redis://localhost:6379/0',
        ttl_seconds=10,
        client=FakeRedisClient(),
    )

    assert store.try_mark_processed('evt-1') is True
    assert store.try_mark_processed('evt-1') is False


def test_is_processed_and_clear():
    store = RedisIdempotencyStore(
        redis_url='redis://localhost:6379/0',
        ttl_seconds=10,
        client=FakeRedisClient(),
    )
    assert store.is_processed('evt-2') is False
    assert store.mark_processed('evt-2') is True
    assert store.is_processed('evt-2') is True
    store.clear('evt-2')
    assert store.is_processed('evt-2') is False
