import torch
from dataclasses import dataclass

@dataclass
class Softmax:
    alpha: float = 0.5
    beta: float = 1.0
    tau: float = 1.0
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Flatten y's spatial dimensions, softmax, and reshape
        c, h, w = y.shape[1:]
        y = y.view(y.shape[0], -1)

        # Convert y to [0, 1] using softmax
        y = torch.softmax(y / self.tau, dim=-1)

        # Reshape
        y = y.view(y.shape[0], c, h, w)

        # Adjust range to [alpha, beta]
        y = self.alpha + (self.beta - self.alpha) * y

        # Multiply
        return x * y
    
    def __str__(self):
        return f"Multiply(alpha={self.alpha},beta={self.beta},tau={self.tau})"