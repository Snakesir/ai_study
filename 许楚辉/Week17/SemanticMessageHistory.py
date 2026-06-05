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
    simple_text_score,
    stable_hash,
    to_vector_array,
    utc_ts,
)


class SemanticMessageHistory:
    """Session-scoped message history with recent and relevant retrieval."""

    def __init__(
        self,
        name: str,
        ttl: int = 3600 * 24,
        redis_url: str = "localhost",
        redis_port: int = 6379,
        redis_password: str = None,
        embedding_method: Optional[Callable[[Union[str, List[str]]], Any]] = None,
        redis_client: Optional[Any] = None,
    ):
        self.name = name
        self.redis = redis_client or redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
        )
        self.ttl = ttl
        self.embedding_method = embedding_method

    @property
    def _ids_key(self) -> str:
        return f"semantic_history:{self.name}:ids"

    def _message_key(self, message_id: str) -> str:
        return f"semantic_history:{self.name}:message:{message_id}"

    def _load_ids(self) -> List[str]:
        ids = self.redis.lrange(self._ids_key, 0, -1)
        normalized = []
        seen = set()
        for message_id in ids:
            if isinstance(message_id, bytes):
                message_id = message_id.decode("utf-8")
            if message_id not in seen:
                normalized.append(message_id)
                seen.add(message_id)
        return normalized

    def _load_messages(self) -> List[Dict[str, Any]]:
        messages = []
        for message_id in self._load_ids():
            message = decode_json(self.redis.get(self._message_key(message_id)))
            if message is not None:
                messages.append(message)
        return messages

    def _normalize_message(self, message: Dict[Any, Any], index: int = 0) -> Dict[str, Any]:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        timestamp = float(message.get("timestamp", utc_ts()))
        metadata = dict(message.get("metadata", {}))
        message_id = message.get("id")
        if message_id is None:
            message_id = stable_hash(f"{self.name}:{role}:{content}:{timestamp}:{index}")
        normalized = {
            "id": message_id,
            "session": self.name,
            "role": role,
            "content": content,
            "metadata": metadata,
            "timestamp": timestamp,
        }
        if self.embedding_method and content:
            normalized["vector"] = to_vector_array(self.embedding_method(content))[0].tolist()
        elif "vector" in message:
            normalized["vector"] = np.asarray(message["vector"], dtype=np.float32).reshape(-1).tolist()
        return normalized

    def add_message(self, message: Union[Dict[Any, Any], List[Dict[Any, Any]]]) -> Union[str, List[str]]:
        if isinstance(message, list):
            return self.add_messages(message)
        return self.add_messages([message])[0]

    def add_messages(self, messages: List[Dict[Any, Any]]) -> List[str]:
        message_ids = []
        for index, message in enumerate(messages):
            normalized = self._normalize_message(message, index=index)
            message_ids.append(normalized["id"])
            self.redis.setex(self._message_key(normalized["id"]), self.ttl, encode_json(normalized))
            self.redis.rpush(self._ids_key, normalized["id"])
        return message_ids

    def store(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        return self.add_messages(
            [
                {"role": "user", "content": prompt, "metadata": metadata or {}},
                {"role": "llm", "content": response, "metadata": metadata or {}},
            ]
        )

    def get_history(self) -> List[Dict[str, Any]]:
        return self._load_messages()

    def get_recent(self, role: Optional[Union[str, List[str]]] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        messages = self.get_history()
        if role:
            roles = set(ensure_list(role))
            messages = [message for message in messages if message.get("role") in roles]
        return messages[-top_k:] if top_k else messages

    def get_relevant(
        self,
        content: str,
        top_k: int = 10,
        role: Optional[Union[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        messages = self.get_history()
        if role:
            roles = set(ensure_list(role))
            messages = [message for message in messages if message.get("role") in roles]
        if not messages:
            return []

        scored = []
        if self.embedding_method:
            query_vector = to_vector_array(self.embedding_method(content))[0]
            for message in messages:
                vector = message.get("vector")
                if vector is None:
                    vector = to_vector_array(self.embedding_method(message.get("content", "")))[0]
                distance = l2_distance(query_vector, vector)
                scored.append((distance, message))
            scored.sort(key=lambda item: item[0])
        else:
            for message in messages:
                score = simple_text_score(content, message.get("content", ""))
                scored.append((1.0 - score, message))
            scored.sort(key=lambda item: item[0])

        return [message for _, message in scored[:top_k]]

    def count(self, role: Optional[Union[str, List[str]]] = None) -> int:
        return len(self.get_recent(role=role, top_k=0))

    def delete_history(self, top_k: int = 10) -> int:
        messages = self.get_history()
        keep = messages[-top_k:] if top_k else []
        self.clear_history()
        if keep:
            self.add_messages(keep)
        return len(keep)

    def drop(self) -> int:
        return self.clear_history()

    def clear_history(self) -> int:
        keys = [self._message_key(message_id) for message_id in self._load_ids()]
        keys.append(self._ids_key)
        return int(self.redis.delete(*keys))


if __name__ == "__main__":
    history = SemanticMessageHistory(name="my-session", redis_url="localhost")
    history.clear_history()
    history.add_messages(
        [
            {"role": "user", "content": "hello, how are you?"},
            {"role": "llm", "content": "I'm doing fine, thanks."},
            {"role": "user", "content": "what is the weather going to be today?"},
        ]
    )
    print("get_history", history.get_history())
    print("get_recent top_k=1", history.get_recent(top_k=1))
