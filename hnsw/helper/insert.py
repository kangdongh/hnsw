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

    def insert(self, vector: NDArray, ef: int):
        # Step 1: Add the vector to the vector storage
        new_id = len(self.vector_storage.vectors)
        self.vector_storage.add_vector(vector)

        # Step 2: Determine the entry point
        entry_point_id = self.layers[-1].entry_id if self.layers[-1].entry_id else new_id

        # Step 3: Insert the node into each layer
        for layer_index in reversed(range(len(self.layers))):
            layer = self.layers[layer_index]
            layer.add_node(new_id)

            # Use LayerSearcher to find the nearest neighbors
            layer_searcher = BaseLayerSearchHelper(layer, self.vector_storage)
            layer_searcher.search(entry_point_id, ef, vector)

            # Connect the new node to its nearest neighbors
            for neighbor in layer_searcher.nn:
                layer.add_edge(new_id, neighbor.vector_id)

            # Update the entry point for the next layer
            if layer_index > 0:
                entry_point_id = layer_searcher.nn[0].vector_id

        # Step 4: Set the entry node for the top layer if it's the first node
        if self.layers[-1].entry_id is None:
            self.layers[-1].set_entry_id(new_id)
