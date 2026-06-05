import fnmatch
import hashlib
import json
import time
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def utc_ts() -> float:
    return time.time()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def to_vector_array(vectors: Any) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def encode_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def decode_json(payload: Any) -> Optional[dict]:
    if payload is None:
        return None
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(payload)


def l2_distance(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=np.float32)
    right_arr = np.asarray(right, dtype=np.float32)
    return float(np.sum((left_arr - right_arr) ** 2))


def simple_text_score(query: str, content: str) -> float:
    query_chars = set(query.lower())
    content_chars = set(content.lower())
    if not query_chars or not content_chars:
        return 0.0
    return len(query_chars & content_chars) / len(query_chars | content_chars)


class NumpyL2Index:
    def __init__(self, dims: int):
        self.dims = dims
        self.vectors = np.empty((0, dims), dtype=np.float32)

    def add(self, vectors: Any) -> None:
        vectors = to_vector_array(vectors)
        if vectors.shape[1] != self.dims:
            raise ValueError(f"Expected vectors with {self.dims} dims, got {vectors.shape[1]}")
        self.vectors = np.vstack([self.vectors, vectors])

    def reset(self) -> None:
        self.vectors = np.empty((0, self.dims), dtype=np.float32)

    def search(self, vectors: Any, k: int) -> Tuple[np.ndarray, np.ndarray]:
        vectors = to_vector_array(vectors)
        if len(self.vectors) == 0:
            return np.empty((len(vectors), 0), dtype=np.float32), np.empty((len(vectors), 0), dtype=np.int64)
        distances = ((vectors[:, None, :] - self.vectors[None, :, :]) ** 2).sum(axis=2)
        order = np.argsort(distances, axis=1)[:, :k]
        sorted_distances = np.take_along_axis(distances, order, axis=1)
        return sorted_distances.astype(np.float32), order.astype(np.int64)


class MemoryPipeline:
    def __init__(self, redis_client: "MemoryRedis"):
        self.redis = redis_client
        self.ops = []

    def __enter__(self) -> "MemoryPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def setex(self, key: str, ttl: int, value: Any) -> "MemoryPipeline":
        self.ops.append(("setex", key, ttl, value))
        return self

    def set(self, key: str, value: Any) -> "MemoryPipeline":
        self.ops.append(("set", key, value))
        return self

    def rpush(self, key: str, value: Any) -> "MemoryPipeline":
        self.ops.append(("rpush", key, value))
        return self

    def lpush(self, key: str, value: Any) -> "MemoryPipeline":
        self.ops.append(("lpush", key, value))
        return self

    def execute(self) -> list:
        results = []
        for op in self.ops:
            if op[0] == "setex":
                results.append(self.redis.setex(op[1], op[2], op[3]))
            elif op[0] == "set":
                results.append(self.redis.set(op[1], op[2]))
            elif op[0] == "rpush":
                results.append(self.redis.rpush(op[1], op[2]))
            elif op[0] == "lpush":
                results.append(self.redis.lpush(op[1], op[2]))
        self.ops = []
        return results


class MemoryRedis:
    def __init__(self, *args, **kwargs):
        self.store = {}
        self.lists = {}

    def pipeline(self) -> MemoryPipeline:
        return MemoryPipeline(self)

    def setex(self, key: str, ttl: int, value: Any) -> bool:
        self.store[key] = value
        return True

    def set(self, key: str, value: Any) -> bool:
        self.store[key] = value
        return True

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def exists(self, key: str) -> int:
        return 1 if key in self.store or key in self.lists else 0

    def delete(self, *keys: Iterable[str]) -> int:
        count = 0
        for key in keys:
            if isinstance(key, (list, tuple)):
                count += self.delete(*key)
                continue
            key = key.decode("utf-8") if isinstance(key, bytes) else key
            if key in self.store:
                del self.store[key]
                count += 1
            if key in self.lists:
                del self.lists[key]
                count += 1
        return count

    def mget(self, keys=None, *extra_keys) -> list:
        if keys is None:
            keys = []
        if isinstance(keys, (str, bytes)):
            key_list = [keys, *extra_keys]
        else:
            key_list = list(keys) + list(extra_keys)
        return [self.get(key.decode("utf-8") if isinstance(key, bytes) else key) for key in key_list]

    def rpush(self, key: str, value: Any) -> int:
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    def lpush(self, key: str, value: Any) -> int:
        self.lists.setdefault(key, []).insert(0, value)
        return len(self.lists[key])

    def lrange(self, key: str, start: int, end: int) -> list:
        items = self.lists.get(key, [])
        stop = None if end == -1 else end + 1
        return items[start:stop]

    def keys(self, pattern: str) -> list:
        all_keys = list(self.store.keys()) + list(self.lists.keys())
        return [key for key in all_keys if fnmatch.fnmatch(key, pattern)]
