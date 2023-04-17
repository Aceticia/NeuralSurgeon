import torch
import torch.nn as nn

from torchvision.models import resnet18, resnet50

from collections import OrderedDict

from src.models.components.subspace_net import SubspaceNet, LayerMetadata

class ResNetModel(SubspaceNet):
    def __init__(self, subspace_size: int, resnet_model: nn.Module) -> None:
        super().__init__(subspace_size=subspace_size)

        # Get the stem layers of the model
        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool

        # Get the layers of the model
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4

        # Freeze the conv
        for m in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in m.parameters():
                param.requires_grad = False

    def layer_forward(self, x: torch.Tensor, layer: str) -> torch.Tensor:
        if layer == "layer1":
            return self.layer1(x)
        elif layer == "layer2":
            return self.layer2(x)
        elif layer == "layer3":
            return self.layer3(x)
        elif layer == "layer4":
            return self.layer4(x)
        else:
            raise ValueError(f"Invalid layer: {layer}")

    def stem_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.maxpool(x)


class ResNet18Model(ResNetModel):
    def __init__(self, subspace_size: int) -> None:
        super().__init__(
            subspace_size=subspace_size,
            resnet_model=resnet18(weights="IMAGENET1K_V1")
        )

    def layer_sizes(self) -> OrderedDict[str, LayerMetadata]:
        return OrderedDict(
            layer1=LayerMetadata(8, 64),
            layer2=LayerMetadata(4, 128),
            layer3=LayerMetadata(2, 256),
            layer4=LayerMetadata(1, 512),
        )

class ResNet50Model(ResNetModel):
    def __init__(self, subspace_size: int) -> None:
        super().__init__(
            subspace_size=subspace_size,
            resnet_model=resnet50(weights="IMAGENET1K_V2")
        )

    def layer_sizes(self) -> OrderedDict[str, LayerMetadata]:
        return OrderedDict(
            layer1=LayerMetadata(8, 64),
            layer2=LayerMetadata(4, 128),
            layer3=LayerMetadata(2, 256),
            layer4=LayerMetadata(1, 512),
        )