"""
Router for neighbors
"""

class NeighborSelfRouter:
    def __init__(self, layer_names, neighbor_up=True, neighbor_skip=0):
        self.layer_names = layer_names
        self.neighbor_up = neighbor_up
        self.neighbor_skip = neighbor_skip

        self.connections = []

        # Return self and a neighbor based on whether up or how many to skip
        for layer_idx, layer in enumerate(layer_names):
            if neighbor_up:
                to_select = layer_idx + neighbor_skip + 1
            else:
                to_select = layer_idx - neighbor_skip - 1

            if to_select >= 0 or to_select < len(layer_names):
                self.connections.append((layer, layer_names[to_select]))

    def __iter__(self):
        yield from self.connections

    def __len__(self):
        return len(self.layer_names)
    
    def __str__(self) -> str:
        return f"{'Up' if self.neighbor_up else 'Down'}NeighborSelfRouter(skip={self.neighbor_skip})"