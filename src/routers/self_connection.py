class AllSelfRouter:
    def __init__(self, layer_names):
        self.layer_names = layer_names

    def __iter__(self):
        yield [(layer, layer) for layer in self.layer_names]

    def __len__(self):
        return 1
    
    def __str__(self) -> str:
        return "AllSelfRouter"