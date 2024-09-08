from typing import Dict, Optional

from hnsw.data.node import Node


class Layer:
    nodes: Dict[int, Node]
    entry_id: int

    def __init__(self):
        self.nodes = {}
        self.entry_id = -1

    def add_node(self, node_id):
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id)

    def remove_node(self, node_id):
        if node_id in self.nodes:
            # Remove edges from other nodes to this node
            for node in self.nodes.values():
                node.remove_edge(node_id)
            # Remove the node itself
            del self.nodes[node_id]

    def add_edge(self, node_id, neighbor_id):
        if node_id in self.nodes and neighbor_id in self.nodes:
            self.nodes[node_id].add_edge(neighbor_id)
            self.nodes[neighbor_id].add_edge(node_id)

    def remove_edge(self, node_id, neighbor_id):
        if node_id in self.nodes and neighbor_id in self.nodes:
            self.nodes[node_id].remove_edge(neighbor_id)
            self.nodes[neighbor_id].remove_edge(node_id)

    def set_entry_id(self, node_id):
        if node_id in self.nodes:
            self.entry_id = node_id

    def get_node(self, node_id):
        return self.nodes.get(node_id)
