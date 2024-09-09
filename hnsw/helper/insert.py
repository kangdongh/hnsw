import random
from typing import List

from numpy.typing import NDArray

from hnsw.data.graph import Graph
from hnsw.data.layer import Layer
from hnsw.data.vector_storage import VectorStorage
from hnsw.helper.base_layer_search import BaseLayerSearchHelper


class InsertHelper:
    layers: List[Layer]
    vector_storage: VectorStorage

    def __init__(self, graph: Graph, vector_storage: VectorStorage):
        self.layers = graph.layers
        self.vector_storage = vector_storage

    def insert(self, ef: int, k: int, vector: NDArray):
        # Step 1: Add the vector to the vector storage
        new_id = len(self.vector_storage.vectors)
        self.vector_storage.add_vector(vector)

        level = random.randrange(len(self.layers)) + 1

        entry_point_id = -1
        for layer_index in reversed(range(level)):
            layer = self.layers[layer_index]
            layer.add_node(new_id)

            # Use LayerSearcher to find the nearest neighbors
            if entry_point_id == -1 and layer.entry_id != -1:
                entry_point_id = layer.entry_id

            if entry_point_id != -1:
                layer_searcher = BaseLayerSearchHelper(layer, self.vector_storage)
                layer_searcher.search(entry_point_id, ef, vector)
                for neighbor in layer_searcher.nn[:k]:
                    layer.add_edge(new_id, neighbor.vector_id)
                entry_point_id = layer_searcher.nn[0].vector_id

            if layer.entry_id == -1:
                layer.entry_id = new_id

