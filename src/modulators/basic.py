import torch
from dataclasses import dataclass

# TODO: Write generalized normalized modulator for the different multiply modulators

@dataclass
class Average:
    alpha: float = 0.5
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + (1-self.alpha) * y
    
    def __str__(self):
        return f"Average(alpha={self.alpha})"


@dataclass
class MultiplyChannelWise:
    alpha: float = 0.75
    beta: float = 1.25

    def __str__(self) -> str:
        return f"MultiplyChannelWise(alpha={self.alpha},beta={self.beta})"

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, c = y.shape[:2]

        # Find magnitude of y
        y = torch.abs(y)

        # Find the max over spatial dimensions
        y_max = y.view(b, c, -1).max(dim=-1)[0]
        y_max = y_max.view(b, c, 1, 1)

        # Normalize y
        y = y / y_max

        # Map y to [alpha, beta]
        y = self.alpha + (self.beta - self.alpha) * y

        # Multiply
        return x * y

@dataclass
class MultiplySpaceWise:
    alpha: float = 0.75
    beta: float = 1.25

    def __str__(self) -> str:
        return f"MultiplySpaceWise(alpha={self.alpha},beta={self.beta})"

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Find magnitude of y
        y = torch.abs(y)

        # Find the max over each feature
        y_max = y.max(dim=1, keepdim=True)[0]

        # Normalize y
        y = y / y_max

        # Map y to [alpha, beta]
        y = self.alpha + (self.beta - self.alpha) * y

        # Multiply
        return x * y


@dataclass
class MultiplyGlobal:
    alpha: float = 0.75
    beta: float = 1.25

    def __str__(self) -> str:
        return f"MultiplyGlobal(alpha={self.alpha},beta={self.beta})"

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Find magnitude of y
        y = torch.abs(y)

        # Find the max over each feature
        y_max = y.flatten(1).max(dim=1, keepdim=True)[0]

        # Reshape to broadcast
        y_max = y_max.view(-1, 1, 1, 1)

        # Normalize y
        y = y / y_max

        # Map y to [alpha, beta]
        y = self.alpha + (self.beta - self.alpha) * y

        # Multiply
        return x * y