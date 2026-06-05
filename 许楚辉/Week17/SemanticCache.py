from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

try:
    import redis
except ModuleNotFoundError:
    from types import SimpleNamespace

    from llm_cache._compat import MemoryRedis

    redis = SimpleNamespace(Redis=MemoryRedis)

from llm_cache._compat import (
    decode_json,
    encode_json,
    ensure_list,
    l2_distance,
    stable_hash,
    to_vector_array,
    utc_ts,
)


class SemanticCache:
    """Redis-backed prompt/response cache with vector similarity lookup."""

    def __init__(
        self,
        name: str,
        embedding_method: Callable[[Union[str, List[str]]], Any],
        ttl: int = 3600 * 24,
        redis_url: str = "localhost",
        redis_port: int = 6379,
        redis_password: str = None,
        distance_threshold: float = 0.1,
        redis_client: Optional[Any] = None,
    ):
        self.name = name
        self.redis = redis_client or redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
        )
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method

    @property
    def _ids_key(self) -> str:
        return f"{self.name}:semantic_cache:ids"

    def _entry_key(self, entry_id: str) -> str:
        return f"{self.name}:semantic_cache:entry:{entry_id}"

    def _make_id(self, prompt: str) -> str:
        return stable_hash(prompt)

    def _load_ids(self) -> List[str]:
        ids = self.redis.lrange(self._ids_key, 0, -1)
        normalized = []
        seen = set()
        for entry_id in ids:
            if isinstance(entry_id, bytes):
                entry_id = entry_id.decode("utf-8")
            if entry_id not in seen:
                normalized.append(entry_id)
                seen.add(entry_id)
        return normalized

    def _load_entries(self) -> List[Dict[str, Any]]:
        entries = []
        for entry_id in self._load_ids():
            entry = decode_json(self.redis.get(self._entry_key(entry_id)))
            if entry is not None:
                entries.append(entry)
        return entries

    def store(
        self,
        prompt: Union[str, List[str]],
        response: Union[str, List[str]],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        filters: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ttl: Optional[int] = None,
    ) -> List[str]:
        prompts = [str(item) for item in ensure_list(prompt)]
        responses = ensure_list(response)
        if len(responses) == 1 and len(prompts) > 1:
            responses = responses * len(prompts)
        if len(prompts) != len(responses):
            raise ValueError("prompt and response must have the same length")

        metadata_list = metadata if isinstance(metadata, list) else [metadata] * len(prompts)
        filters_list = filters if isinstance(filters, list) else [filters] * len(prompts)
        vectors = to_vector_array(self.embedding_method(prompts))
        if vectors.shape[0] != len(prompts):
            raise ValueError("embedding_method must return one vector per prompt")

        entry_ids = []
        for index, item in enumerate(prompts):
            entry_id = self._make_id(item)
            entry_ids.append(entry_id)
            payload = {
                "id": entry_id,
                "prompt": item,
                "response": responses[index],
                "vector": vectors[index].tolist(),
                "metadata": metadata_list[index] or {},
                "filters": filters_list[index] or {},
                "created_at": utc_ts(),
            }
            self.redis.setex(self._entry_key(entry_id), ttl or self.ttl, encode_json(payload))
            self.redis.rpush(self._ids_key, entry_id)
        return entry_ids

    def _filters_match(self, entry_filters: Dict[str, Any], requested: Optional[Dict[str, Any]]) -> bool:
        if not requested:
            return True
        return all(entry_filters.get(key) == value for key, value in requested.items())

    def check(
        self,
        prompt: str,
        top_k: int = 1,
        distance_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        entries = [
            entry for entry in self._load_entries()
            if self._filters_match(entry.get("filters", {}), filters)
        ]
        if not entries:
            return []

        query_vector = to_vector_array(self.embedding_method(prompt))[0]
        threshold = self.distance_threshold if distance_threshold is None else distance_threshold
        hits = []
        for entry in entries:
            distance = l2_distance(query_vector, entry["vector"])
            if distance <= threshold:
                hits.append(
                    {
                        "id": entry["id"],
                        "prompt": entry["prompt"],
                        "response": entry["response"],
                        "distance": distance,
                        "metadata": entry.get("metadata", {}),
                        "filters": entry.get("filters", {}),
                        "created_at": entry.get("created_at"),
                    }
                )
        return sorted(hits, key=lambda item: item["distance"])[:top_k]

    def call(
        self,
        prompt: str,
        top_k: int = 1,
        distance_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return self.check(
            prompt=prompt,
            top_k=top_k,
            distance_threshold=distance_threshold,
            filters=filters,
        )

    def delete_by_id(self, entry_id: str) -> int:
        return int(self.redis.delete(self._entry_key(entry_id)))

    def clear(self) -> int:
        return self.clear_cache()

    def clear_cache(self) -> int:
        keys = [self._entry_key(entry_id) for entry_id in self._load_ids()]
        keys.append(self._ids_key)
        return int(self.redis.delete(*keys))


if __name__ == "__main__":
    def get_embedding(text):
        if isinstance(text, str):
            text = [text]
        return np.array([np.ones(4, dtype=np.float32) for _ in text])

    cache = SemanticCache(
        name="semantic_cache",
        embedding_method=get_embedding,
        ttl=360,
        distance_threshold=0.1,
    )
    cache.clear_cache()
    cache.store(prompt="hello world", response="hello world response")
    print(cache.check(prompt="hello world"))
