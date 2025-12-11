"""Gradient Reversal Layer for Domain Adversarial Neural Networks.

Reference:
    Ganin, Y., & Lempitsky, V. (2015). "Unsupervised domain adaptation by
    backpropagation." International conference on machine learning (pp. 1180-1189).
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient Reversal Layer from 'Unsupervised Domain Adaptation by Backpropagation'.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        self.alpha = alpha


def get_grl_alpha(epoch: int, max_epoch: int, schedule: str = "linear",
                  low: float = 0.0, high: float = 1.0) -> float:
    """
    Calculate the alpha value for GRL based on training progress.
    """
    if schedule == "constant":
        return high
    elif schedule == "linear":
        progress = epoch / max_epoch
        return low + (high - low) * progress
    elif schedule == "exponential":
        # lambda_p = 2/(1+exp(-10*p)) - 1
        # p is the training progress linearly changing from 0 to 1
        progress = epoch / max_epoch
        return high * (2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * progress)).item()) - 1.0)
    else:
        raise ValueError(f"Unknown schedule: {schedule}. Choose from 'constant', 'linear', 'exponential'")
