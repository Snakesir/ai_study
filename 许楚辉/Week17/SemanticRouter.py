from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from llm_cache._compat import ensure_list, l2_distance, to_vector_array


@dataclass
class Route:
    name: str
    references: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    distance_threshold: Optional[float] = None


@dataclass
class RouteMatch:
    name: str
    distance: float
    reference: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticRouter:
    """Semantic intent router inspired by RedisVL's SemanticRouter."""

    def __init__(
        self,
        routes: Optional[List[Route]] = None,
        embedding_method: Optional[Callable[[Union[str, List[str]]], Any]] = None,
        distance_threshold: float = 0.3,
    ):
        self.routes: Dict[str, Route] = {}
        self.embedding_method = embedding_method or self._default_embedding
        self.distance_threshold = distance_threshold
        if routes:
            for route in routes:
                self.routes[route.name] = route

    def _default_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        texts = ensure_list(text)
        vectors = []
        for item in texts:
            vector = np.zeros(64, dtype=np.float32)
            for char in item.lower():
                vector[ord(char) % 64] += 1.0
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.vstack(vectors)

    def add_route(
        self,
        questions: Union[List[str], str],
        target: str,
        metadata: Optional[Dict[str, Any]] = None,
        distance_threshold: Optional[float] = None,
    ) -> Route:
        route = Route(
            name=target,
            references=[str(question) for question in ensure_list(questions)],
            metadata=metadata or {},
            distance_threshold=distance_threshold,
        )
        self.routes[target] = route
        return route

    def remove_route(self, name: str) -> bool:
        return self.routes.pop(name, None) is not None

    def add_route_references(self, name: str, references: Union[List[str], str]) -> Route:
        if name not in self.routes:
            raise KeyError(f"Route {name!r} does not exist")
        self.routes[name].references.extend(str(reference) for reference in ensure_list(references))
        return self.routes[name]

    def _reference_matches(self, question: str) -> List[RouteMatch]:
        if not self.routes:
            return []
        query_vector = to_vector_array(self.embedding_method(question))[0]
        matches = []
        for route in self.routes.values():
            threshold = route.distance_threshold
            if threshold is None:
                threshold = self.distance_threshold
            for reference in route.references:
                ref_vector = to_vector_array(self.embedding_method(reference))[0]
                distance = l2_distance(query_vector, ref_vector)
                if distance <= threshold:
                    matches.append(
                        RouteMatch(
                            name=route.name,
                            distance=distance,
                            reference=reference,
                            metadata=dict(route.metadata),
                        )
                    )
        return sorted(matches, key=lambda item: item.distance)

    def route(self, question: str) -> Optional[RouteMatch]:
        matches = self._reference_matches(question)
        return matches[0] if matches else None

    def route_many(self, question: str, top_k: int = 3) -> List[RouteMatch]:
        return self._reference_matches(question)[:top_k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "distance_threshold": self.distance_threshold,
            "routes": [
                {
                    "name": route.name,
                    "references": route.references,
                    "metadata": route.metadata,
                    "distance_threshold": route.distance_threshold,
                }
                for route in self.routes.values()
            ],
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        embedding_method: Optional[Callable[[Union[str, List[str]]], Any]] = None,
    ) -> "SemanticRouter":
        routes = [
            Route(
                name=route["name"],
                references=list(route.get("references", [])),
                metadata=dict(route.get("metadata", {})),
                distance_threshold=route.get("distance_threshold"),
            )
            for route in data.get("routes", [])
        ]
        return cls(
            routes=routes,
            embedding_method=embedding_method,
            distance_threshold=data.get("distance_threshold", 0.3),
        )


if __name__ == "__main__":
    router = SemanticRouter()
    router.add_route(["Hi, good morning", "Hi, good afternoon"], target="greeting")
    router.add_route(["如何退货"], target="refund")
    print(router.route("Hi, good morning"))
