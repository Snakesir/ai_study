from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import redis
except ModuleNotFoundError:
    from types import SimpleNamespace

    from llm_cache._compat import MemoryRedis

    redis = SimpleNamespace(Redis=MemoryRedis)

from llm_cache._compat import decode_json, encode_json, ensure_list, stable_hash, utc_ts


class EmbeddingsCache:
    """Cache exact text embeddings by text hash and model name.

    This teaching version mirrors RedisVL's embedding cache idea without
    depending on RediSearch: Redis stores one JSON payload per text/model key.
    """

    def __init__(
        self,
        name: str,
        ttl: int = 3600 * 24,
        redis_url: str = "localhost",
        redis_port: int = 6379,
        redis_password: str = None,
        redis_client: Optional[Any] = None,
    ):
        self.name = name
        self.ttl = ttl
        self.redis = redis_client or redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
        )

    def _key(self, content: str, model_name: str = "default") -> str:
        return f"{self.name}:embedding:{model_name}:{stable_hash(content)}"

    def set(
        self,
        content: str,
        model_name: str,
        embedding: Union[np.ndarray, List[float]],
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        payload = {
            "content": content,
            "model_name": model_name,
            "embedding": vector.tolist(),
            "dtype": "float32",
            "dims": int(vector.shape[0]),
            "metadata": metadata or {},
            "created_at": utc_ts(),
        }
        return bool(self.redis.setex(self._key(content, model_name), ttl or self.ttl, encode_json(payload)))

    def get(self, content: str, model_name: str = "default") -> Optional[Dict[str, Any]]:
        payload = decode_json(self.redis.get(self._key(content, model_name)))
        if payload is None:
            return None
        payload["embedding"] = np.asarray(payload["embedding"], dtype=np.float32)
        return payload

    def mset(
        self,
        records: List[Dict[str, Any]],
        default_model_name: str = "default",
        ttl: Optional[int] = None,
    ) -> List[bool]:
        results = []
        for record in records:
            results.append(
                self.set(
                    content=record["content"],
                    model_name=record.get("model_name", default_model_name),
                    embedding=record["embedding"],
                    metadata=record.get("metadata"),
                    ttl=ttl,
                )
            )
        return results

    def mget(self, contents: List[str], model_name: str = "default") -> List[Optional[Dict[str, Any]]]:
        return [self.get(content, model_name=model_name) for content in contents]

    def exists(self, content: str, model_name: str = "default") -> bool:
        return bool(self.redis.exists(self._key(content, model_name)))

    def drop(self, content: Union[List[str], str], model_name: str = "default") -> int:
        keys = [self._key(item, model_name) for item in ensure_list(content)]
        if not keys:
            return 0
        return int(self.redis.delete(*keys))

    def clear(self) -> int:
        keys = self.redis.keys(f"{self.name}:embedding:*")
        if not keys:
            return 0
        return int(self.redis.delete(*keys))

    def store(
        self,
        text: Union[List[str], str],
        embedding: Union[np.ndarray, List[List[float]], List[float]],
        model_name: str = "default",
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> List[bool]:
        texts = ensure_list(text)
        vectors = np.asarray(embedding, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        metadata_list = ensure_list(metadata) if isinstance(metadata, list) else [metadata] * len(texts)
        return [
            self.set(item, model_name=model_name, embedding=vectors[index], metadata=metadata_list[index])
            for index, item in enumerate(texts)
        ]

    def call(self, text: Union[List[str], str], model_name: str = "default") -> List[Optional[np.ndarray]]:
        return [
            None if record is None else record["embedding"]
            for record in self.mget(ensure_list(text), model_name=model_name)
        ]

    def delete(self, text: Union[List[str], str], model_name: str = "default") -> int:
        return self.drop(text, model_name=model_name)


if __name__ == "__main__":
    embed_cache = EmbeddingsCache(name="embedding_cache", ttl=360)
    vector = np.random.rand(768).astype(np.float32)
    print(embed_cache.store(text="hello world", embedding=vector))
    print(embed_cache.call(text="hello world"))
    print(embed_cache.delete(text="hello world"))
