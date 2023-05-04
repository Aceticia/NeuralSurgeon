"""
Router for concatenating multiple routers
"""

class ConcatRouter:
    def __init__(self, layer_names, *args):
        self.routers = [router_factory(layer_names) for router_factory in args]

        # Assert lengths are either 1 or equal to max size
        self.max_size = max(len(router) for router in self.routers)
        for router in self.routers:
            assert len(router) in [1, self.max_size]

        # Construct the connections.
        connections = []
        for router in self.routers:
            connections_curr_router = list(router)
            if len(router) == 1:
                connections_curr_router *= self.max_size
            connections.extend(connections_curr_router)
        
        # Invert the order of nested list
        connections = list(zip(*connections))
        self.connections = connections

    def __iter__(self):
        yield from self.connections

    def __len__(self):
        return len(self.connections)

    def __str__(self) -> str:
        return f"ConcatRouter({','.join([str(router) for router in self.routers])})"