"""
Parent class of all the morphable convolution nets. 

There is a catalog of each layer of the network. When running forward,
we can choose to condition each layer on some past activations. For example,
the layer 1 can have influence from layer 3, and also layer 3 from layer 2.

During the forward pass, we can also optionally skip layers.

But doing this is not trivial. Target and source layers might not have the same spatial
size and feature size. There are 2 cases:

1. Source feature comes from an earlier layer then target. This is a simple case. We can
generate the corresponding target feature by ignoring the convolution ops between
the source and the target layer, only applying the downsampling layers and transition
layers. This generates a new feature map that is guaranteed to be the same size as the
target feature map.

2. Source feature comes from a later layer then target. In this case, we need to
"invert" the convolution operation of many layers. We first map the layers to a
common subspace with one encoder per layer, and use a decoder per layer to map from 
the common subspace to the target layer.

There are two choices about the subspace. We can try to use a subspace that has a
spatial dimension [H, W, D], with a specified size. It should be a very small size.

When generating the target feature, we first generate features of size [H, W, D_target]
and then upsample them to the target size.

TODO:
1. Allow specification of loss. Now we only use triplet loss with L2 distance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from random import sample

from typing import Dict, List, Tuple
from dataclasses import dataclass

from collections import OrderedDict

from src.models.components.layer_norm import LayerNorm

@dataclass
class LayerMetadata:
    spatial_size: int
    num_channels: int

class SubspaceNet(nn.Module):
    def __init__(self, subspace_size: int) -> None:
        super().__init__()

        # First get the layer sizes
        self.layer_meta_data = self.layer_sizes()

        # Resolve the spatial size of the common subspace
        # We use the smallest spatial size of all the layers
        self.subspace_spatial_size = min(
            meta.spatial_size for meta in self.layer_meta_data.values()
        )

        # Then create the subspace encoders and decoders
        self.subspace_encoders = nn.ModuleDict()
        self.subspace_decoders = nn.ModuleDict()

        # Create the subspace encoders
        for layer, meta in self.layer_meta_data.items():
            self.subspace_encoders[layer] = self.create_encoder(meta, subspace_size)

        # Create the subspace decoders
        for layer, meta in self.layer_meta_data.items():
            self.subspace_decoders[layer] = self.create_decoder(meta, subspace_size)

        # Learn a spatial affine normalization for the subspace
        self.norm = LayerNorm(subspace_size, data_format="channels_first")

    def create_encoder(self, meta: LayerMetadata, subspace_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(meta.num_channels, subspace_size, kernel_size=1),
            LayerNorm(subspace_size, data_format="channels_first"),
            nn.ReLU(),
            nn.Conv2d(subspace_size, subspace_size, kernel_size=1),
            nn.AdaptiveAvgPool2d(self.subspace_spatial_size)
        )

    def create_decoder(self, meta: LayerMetadata, subspace_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(subspace_size, subspace_size, kernel_size=1),
            LayerNorm(subspace_size, data_format="channels_first"),
            nn.ReLU(),
            nn.Conv2d(subspace_size, meta.num_channels, kernel_size=1),
            nn.Upsample(size=meta.spatial_size, mode='bilinear', align_corners=True)
        )

    def predict_layer(self, subspace_feat: torch.Tensor, layer: str) -> torch.Tensor:
        return self.subspace_decoders[layer](subspace_feat)

    def to_subspace(self, x: torch.Tensor, layer: str) -> torch.Tensor:
        return self.norm(self.subspace_encoders[layer](x))
    
    def from_to_pred(self, x: torch.Tensor, layer1: str, layer2: str) -> torch.Tensor:
        return self.predict_layer(self.to_subspace(x, layer1), layer2)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: Dict[str, torch.Tensor] = None,
        alpha: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # Stem
        x = self.stem_forward(x)

        # Dict to keep track
        res_dict = OrderedDict()

        # Iterate through layers
        for layer_name in self.layer_meta_data.keys():
            x = self.layer_forward(x, layer_name)

            if conditioning is not None and layer_name in conditioning:
                x = alpha * x + (1-alpha) * conditioning[layer_name]

            res_dict[layer_name] = x

        # Average pool
        x = x.mean(-1).mean(-1)

        return x, res_dict
    
    def conditioned_forward_single(
        self,
        x: torch.Tensor,
        condition_dict: Dict[str, torch.Tensor],
        layer_conditions: List[Tuple[str, str]],
        alpha: float=1.0
    ) -> torch.Tensor:
        # Iterate over the layer conditions and predict the target
        conditionings = {}
        for from_layer, to_layer in layer_conditions:
            pred_to_layer = self.from_to_pred(condition_dict[from_layer], from_layer, to_layer)
            conditionings[to_layer] = pred_to_layer

        # Then condition each layer
        return self(x, conditionings, alpha=alpha)

    def conditioned_forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        layer_conditions: List[Tuple[str, str]],
        alpha: float=1.0
    ) -> torch.Tensor:
        """
        We condition each layer from x to y
        """
        # First forward pass
        _, res_dict = self(x)
        return self.conditioned_forward_single(x, res_dict, layer_conditions, alpha=alpha)

    def get_pairwise_predictions(
        self,
        layer_activations: Dict[str, torch.Tensor],
        from_layer: str,
        to_layer: str
    ) -> torch.Tensor:
        """
        Predict the target layer from the anchor layer.
        """
        subspace_from = self.to_subspace(layer_activations[from_layer], from_layer)
        return self.predict_layer(subspace_from, to_layer)

    def sample_layer_pair_loss(
        self,
        layer_activations: Dict[str, torch.Tensor],
        n_samples: int=10
    ) -> torch.Tensor:
        """
        Triplet loss between two random layers. The anchor is the prediction,
        the positive is the target layer, and the negative is the anchor
        with the batch rolled by 1.
        """

        loss = 0
        for _ in range(n_samples):
            from_layer, to_layer = sample(sorted(layer_activations), 2)

            anchor = self.get_pairwise_predictions(
                layer_activations,
                from_layer,
                to_layer
            )
            positive = layer_activations[to_layer]

            # # Roll batch as negative sample
            # negative = torch.roll(anchor, 1, 0)

            # Flatten the features
            anchor = anchor.flatten(1)
            positive = positive.flatten(1)
            # negative = negative.flatten(1)

            # Compute the loss
            loss += (anchor - positive).abs()

        return loss / n_samples
    
    def get_mapping_params(self):
        return nn.ModuleList([
            self.subspace_encoders,
            self.subspace_decoders,
        ]).parameters()

    def get_final_output_size(self):
        return list(self.layer_meta_data.values())[-1].num_channels

    # The following are to be implemented by the subclass
    def layer_sizes(self) -> OrderedDict[str, LayerMetadata]:
        raise NotImplementedError()

    def stem_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def layer_forward(self, x: torch.Tensor, layer: str) -> torch.Tensor:
        raise NotImplementedError()