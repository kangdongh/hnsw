from typing import Dict

import numpy as np
from numpy.typing import NDArray


class VectorStorage:
    dim: int
    vectors: Dict[int, NDArray]

    def __init__(self, dim):
        self.dim = dim
        self.vectors = dict()

    def add_vector(self, vector: NDArray):
        assert len(vector.shape) == 1
        assert vector.shape[0] == self.dim
        vector_id = len(self.vectors)

        assert vector_id not in self.vectors

        self.vectors[vector_id] = vector

    def get_distance(self, query_vector: NDArray, target_vector_id: int):
        # return cosine distance
        assert target_vector_id in self.vectors, f"Unknown vector id: {target_vector_id}"
        target_vector = self.vectors[target_vector_id]
        inner_prod = np.dot(query_vector, target_vector)
        abs_prod = np.linalg.norm(query_vector) * np.linalg.norm(target_vector)
        cosine_similarity = inner_prod / abs_prod
        cosine_distance = 1 - cosine_similarity
        return cosine_distance
