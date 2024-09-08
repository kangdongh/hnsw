from typing import List

from hnsw.data.layer import Layer


class Graph:
    layers: List[Layer]

    def __init__(self, num_layers):
        self.layers = [Layer() for _ in range(num_layers)]

    def add_node_to_layer(self, layer_index, node_id):
        if 0 <= layer_index < len(self.layers):
            self.layers[layer_index].add_node(node_id)

    def remove_node_from_layer(self, layer_index, node_id):
        if 0 <= layer_index < len(self.layers):
            self.layers[layer_index].remove_node(node_id)

    def add_edge_to_layer(self, layer_index, node_id, neighbor_id):
        if 0 <= layer_index < len(self.layers):
            self.layers[layer_index].add_edge(node_id, neighbor_id)

    def set_entry_node_for_layer(self, layer_index, node_id):
        if 0 <= layer_index < len(self.layers):
            self.layers[layer_index].set_entry_node(node_id)

    def get_layer(self, layer_index):
        if 0 <= layer_index < len(self.layers):
            return self.layers[layer_index]
