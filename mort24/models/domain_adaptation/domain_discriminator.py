"""Domain Discriminator Network for Domain Adversarial Training.
"""

import gin
import torch
import torch.nn as nn
from typing import List


@gin.configurable("DomainDiscriminator")
class DomainDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: List[int] = None,
        num_domains: int = 2,
        dropout: float = 0.3,
        use_batchnorm: bool = False,
    ):
        super(DomainDiscriminator, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_domains = num_domains
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU(inplace=True))

            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_domains))

        self.classifier = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        original_shape = x.shape
        if len(x.shape) == 3:
            # (batch_size, seq_len, feature_dim) -> (batch_size * seq_len, feature_dim)
            batch_size, seq_len, feature_dim = x.shape
            x = x.reshape(-1, feature_dim)
            output = self.classifier(x)
            output = output.reshape(batch_size, seq_len, self.num_domains)
        else:
            # (batch_size, feature_dim)
            output = self.classifier(x)

        return output


@gin.configurable("SimpleDomainDiscriminator")
class SimpleDomainDiscriminator(nn.Module):

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_domains: int = 2,
        dropout: float = 0.2,
    ):
        super(SimpleDomainDiscriminator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_domains),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        original_shape = x.shape
        if len(x.shape) == 3:
            batch_size, seq_len, feature_dim = x.shape
            x = x.reshape(-1, feature_dim)
            output = self.classifier(x)
            output = output.reshape(batch_size, seq_len, self.num_domains)
        else:
            output = self.classifier(x)

        return output
