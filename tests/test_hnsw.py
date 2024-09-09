import random
from unittest import TestCase

import numpy as np

from hnsw.hnsw import HNSW

VECTOR_SIZE = 1024

NUM_INPUTS = 5


class TestHNSW(TestCase):
    def setUp(self):
        self.hnsw = HNSW(10, VECTOR_SIZE)
        random.seed(0)
        np.random.seed(0)

    def test_insert(self):
        self.construct_hnsw()

        self.assertEqual(NUM_INPUTS, len(self.hnsw.vector_storage.vectors))
        self.assertEqual(NUM_INPUTS, len(self.hnsw.graph.layers[0].nodes))

    def construct_hnsw(self):
        ef = 10
        k = 5
        for _ in range(NUM_INPUTS):
            v = np.random.random((VECTOR_SIZE,))
            self.hnsw.insert(ef, k, v)
        for layer in self.hnsw.graph.layers:
            print(len(layer.nodes))

    def test_search(self):
        self.construct_hnsw()

        for vid, vector in self.hnsw.vector_storage.vectors.items():
            nn = self.hnsw.search(5, vector)
            self.assertEqual(nn[0], vid)
