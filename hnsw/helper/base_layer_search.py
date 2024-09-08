import bisect
import heapq
from dataclasses import dataclass
from typing import Set, List

from numpy.typing import NDArray

from hnsw.data.layer import Layer
from hnsw.data.vector_storage import VectorStorage


@dataclass
class SearchNode:
    distance: float = 0.0
    vector_id: int = -1

    def __lt__(self, other):
        return self.distance < other.distance


class BaseLayerSearchHelper:
    layer: Layer
    vector_storage: VectorStorage
    visited: Set[int]
    candidates: List[SearchNode]
    nn: List[SearchNode]

    def __init__(self, layer: Layer, vector_storage: VectorStorage):
        self.layer = layer
        self.vector_storage = vector_storage
        self.visited = set()
        self.candidates = []
        self.nn = []

    def get_search_node(self, query_vector, target_vector_id):
        self.visited.add(target_vector_id)
        distance = self.vector_storage.get_distance(query_vector, target_vector_id)
        return SearchNode(distance, target_vector_id)

    def update_node(self, node: SearchNode, ef):
        if len(self.nn) >= ef and self.nn[-1].distance < node.distance:
            return

        heapq.heappush(self.candidates, node)
        bisect.insort(self.nn, node)
        self.nn = self.nn[:ef]

    def search(self, entry_id: int, ef: int, vector: NDArray):
        init_node = self.get_search_node(vector, entry_id)
        self.update_node(init_node, ef)
        self.search_(ef, vector)

    def search_(self, ef: int, vector: NDArray):
        if not self.candidates:
            return

        current_id = heapq.heappop(self.candidates).vector_id
        for neighbor_id in self.layer.get_node(current_id).get_neighbors():
            if neighbor_id in self.visited:
                continue
            self.update_node(self.get_search_node(vector, neighbor_id), ef)

        self.search_(ef, vector)
