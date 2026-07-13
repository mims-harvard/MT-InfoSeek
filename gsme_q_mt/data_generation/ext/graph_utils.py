from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Dict, Iterable, List, Set, Tuple


try:  # pragma: no cover
    import networkx as nx  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class SimpleDiGraph:
        def __init__(self) -> None:
            self._succ: Dict[str, Set[str]] = {}
            self._pred: Dict[str, Set[str]] = {}

        def add_node(self, node: str) -> None:
            self._succ.setdefault(node, set())
            self._pred.setdefault(node, set())

        def add_edge(self, source: str, target: str) -> None:
            self.add_node(source)
            self.add_node(target)
            self._succ[source].add(target)
            self._pred[target].add(source)

        @property
        def nodes(self) -> List[str]:
            return list(self._succ.keys())

        @property
        def edges(self) -> List[Tuple[str, str]]:
            return [
                (source, target)
                for source, targets in self._succ.items()
                for target in targets
            ]

        def in_degree(self, node: str) -> int:
            return len(self._pred.get(node, set()))

        def number_of_nodes(self) -> int:
            return len(self._succ)

    def ancestors(graph: SimpleDiGraph, node: str) -> Set[str]:
        seen: Set[str] = set()
        stack = list(graph._pred.get(node, set()))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(graph._pred.get(current, set()))
        return seen

    def is_directed_acyclic_graph(graph: SimpleDiGraph) -> bool:
        indegree = {node: graph.in_degree(node) for node in graph.nodes}
        queue = deque(node for node, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            current = queue.popleft()
            visited += 1
            for nxt in graph._succ.get(current, set()):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        return visited == len(graph.nodes)

    nx = SimpleNamespace(
        DiGraph=SimpleDiGraph,
        ancestors=ancestors,
        is_directed_acyclic_graph=is_directed_acyclic_graph,
    )
