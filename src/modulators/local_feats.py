import torch
from dataclasses import dataclass

# TODO: Write generalized normalized modulator for the different multiply modulators
# Guideline for alpha: It should be 0 when no effect of priming, and 1 when the effect is maximum

@dataclass
class Average:
    alpha: float = 0.5
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (1-self.alpha) * x + self.alpha * y
    
    def __str__(self):
        return f"Average(alpha={self.alpha})"


@dataclass
class MultiplyChannelWise:
    alpha: float = 0.5

    def __str__(self) -> str:
        return f"MultiplyChannelWise(alpha={self.alpha})"

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, c = y.shape[:2]

        # Find magnitude of y
        y = torch.abs(y)

        # Find the max over spatial dimensions
        y_max = y.view(b, c, -1).max(dim=-1)[0]
        y_max = y_max.view(b, c, 1, 1)

        # Normalize y
        y = y / y_max

        # Map y to [1-alpha, 1+alpha]
        y = 1 + self.alpha * (2*y - 1)

        # Multiply
        return x * y