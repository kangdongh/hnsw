from typing import List

from numpy.typing import NDArray

from hnsw.data.graph import Graph
from hnsw.data.vector_storage import VectorStorage
from hnsw.helper.base_layer_search import BaseLayerSearchHelper


class SearchHelper:
    graph: Graph
    vector_storage: VectorStorage

    def __init__(self, graph: Graph, vector_storage: VectorStorage):
        self.graph = graph
        self.vector_storage = vector_storage

    def search(self, ef: int, vector: NDArray):
        search_result = [self.graph.layers[-1].entry_id]
        for layer in reversed(self.graph.layers):
            entry_point = search_result[0]
            search_result = self.search_layer(layer, entry_point, ef, vector)

        return search_result

    def search_layer(self, layer, entry_id, ef, vector) -> List[int]:
        layer_searcher = BaseLayerSearchHelper(layer, self.vector_storage)
        layer_searcher.search(entry_id, ef, vector)
        return [n.vector_id for n in layer_searcher.nn]
