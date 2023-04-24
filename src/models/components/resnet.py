import torch
import torch.nn as nn

from torchvision.models import resnet18, resnet34

from collections import OrderedDict

from src.models.components.subspace_net import SubspaceNet, LayerMetadata

class ResNetModel(SubspaceNet):
    constructor = None
    def __init__(self, subspace_size: int) -> None:
        super().__init__(subspace_size=subspace_size)

        # Get the model
        resnet_model = self.constructor(weights=self.weights)

        # Get the stem layers of the model
        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool

        modules = OrderedDict()

        # Get the internal blocks in resnet
        for layer_idx in range(4):
            layer_name = f"layer{layer_idx + 1}"

            # Get layer from resnet
            layer = getattr(resnet_model, layer_name)

            # Iterate over the blocks in the layer
            for block_idx, block in enumerate(layer):
                modules[f"{layer_name}-block{block_idx}"] = block

        self.blocks = nn.ModuleDict(modules)

        # Freeze the conv
        for m in [self.conv1, self.bn1, self.blocks]:
            for param in m.parameters():
                param.requires_grad = False

    def layer_forward(self, x: torch.Tensor, layer: str) -> torch.Tensor:
        return self.blocks[layer](x)

    def stem_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.maxpool(x)

    def layer_sizes(self) -> OrderedDict[str, LayerMetadata]:
        sizes = [(32,64), (16,128), (8,256), (4,512)]
        res_dict = OrderedDict()
        for layer_idx in range(4):
            for block_idx in range(self.num_blocks[layer_idx]):
                block_key = f"layer{layer_idx+1}-block{block_idx}"
                res_dict[block_key] = LayerMetadata(*sizes[layer_idx])
        return res_dict


class ResNet18Model(ResNetModel):
    constructor = resnet18
    weights = "IMAGENET1K_V1"
    num_blocks = [2, 2, 2, 2]


class ResNet34Model(ResNetModel):
    constructor = resnet34
    weights = "IMAGENET1K_V1"
    num_blocks = [3, 4, 6, 3]