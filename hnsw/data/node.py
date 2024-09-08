from typing import Set


class Node:
    id: int
    neighbors: Set[int]

    def __init__(self, node_id):
        self.id = node_id
        self.neighbors = set()

    def add_edge(self, neighbor_id):
        self.neighbors.add(neighbor_id)

    def remove_edge(self, neighbor_id):
        self.neighbors.discard(neighbor_id)

    def get_neighbors(self):
        return self.neighbors
