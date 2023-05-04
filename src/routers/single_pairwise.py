"""
Pairwise router that connects each layer to each other layer.
"""

from itertools import product

class SinglePairwiseRouter:
    def __init__(self, layer_names):
        self.layer_names = layer_names

    def __iter__(self):
        for layer_from, layer_to in product(self.layer_names, repeat=2):
            yield [(layer_from, layer_to)]

    def __len__(self):
        return len(self.layer_names) ** 2

    def __str__(self) -> str:
        return "SinglePairwiseRouter"