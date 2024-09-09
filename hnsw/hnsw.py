from numpy.typing import NDArray

from hnsw.data.graph import Graph
from hnsw.data.vector_storage import VectorStorage
from hnsw.helper.insert import InsertHelper
from hnsw.helper.search import SearchHelper


class HNSW:
    graph: Graph
    vector_storage: VectorStorage

    def __init__(self, num_layers: int, vector_size: int):
        self.graph = Graph(num_layers)
        self.vector_storage = VectorStorage(vector_size)
        self._search_helper = SearchHelper(self.graph, self.vector_storage)
        self._insert_helper = InsertHelper(self.graph, self.vector_storage)

    def insert(self, ef: int, k: int, vector: NDArray):
        self._insert_helper.insert(ef, k, vector)

    def search(self, ef: int, vector: NDArray):
        return self._search_helper.search(ef, vector)
