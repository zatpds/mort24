"""Domain Adversarial Neural Network (DANN) implementation for ICU predictions.

Reference:
    Ganin, Y., et al. (2016). "Domain-adversarial training of neural networks."
    The Journal of Machine Learning Research, 17(1), 2096-2030.
"""

import gin
import logging
import torch.nn as nn
from typing import Optional

from mort24.constants import RunMode
from mort24.models.dl_models.layers import PositionalEncoding, TransformerBlock
from mort24.models.wrappers import DLPredictionWrapper
from mort24.models.domain_adaptation.gradient_reversal import GradientReversalLayer
from mort24.models.domain_adaptation.domain_discriminator import DomainDiscriminator


@gin.configurable("DANNTransformer")
class DANNTransformer(DLPredictionWrapper):

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        hidden: int = 128,
        heads: int = 4,
        ff_hidden_mult: int = 4,
        depth: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        dropout_att: float = 0.1,
        pos_encoding: bool = True,
        l1_reg: float = 0,
        discriminator_hidden_dims: list = None,
        discriminator_dropout: float = 0.3,
        grl_alpha: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        hidden = hidden if hidden % 2 == 0 else hidden + 1

        self.hidden = hidden
        self.num_classes = num_classes
        self.l1_reg = l1_reg
        self.grl_alpha = grl_alpha

        self.input_embedding = nn.Linear(input_size[2], hidden)

        if pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden)
        else:
            self.pos_encoder = None

        # transformer blocks
        t_blocks = []
        for _ in range(depth):
            t_blocks.append(
                TransformerBlock(
                    emb=hidden,
                    hidden=hidden,
                    heads=heads,
                    mask=True,
                    ff_hidden_mult=ff_hidden_mult,
                    dropout=dropout,
                    dropout_att=dropout_att,
                )
            )
        self.t_blocks = nn.Sequential(*t_blocks)

        self.task_classifier = nn.Linear(hidden, num_classes)

        self.grl = GradientReversalLayer(alpha=grl_alpha)

        if discriminator_hidden_dims is None:
            discriminator_hidden_dims = [256, 128]

        self.domain_discriminator = DomainDiscriminator(
            input_dim=hidden,
            hidden_dims=discriminator_hidden_dims,
            num_domains=2, 
            dropout=discriminator_dropout,
        )

        logging.info(f"Initialized DANN Transformer with hidden={hidden}, depth={depth}, heads={heads}")

    def extract_features(self, x):

        x = self.input_embedding(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.t_blocks(x)
        return x

    def forward(self, x, alpha: Optional[float] = None, return_domain_output: bool = False):

        features = self.extract_features(x)

        task_output = self.task_classifier(features)

        if not return_domain_output:
            return task_output

        if alpha is not None:
            self.grl.set_alpha(alpha)

        reversed_features = self.grl(features)
        domain_output = self.domain_discriminator(reversed_features)

        return task_output, domain_output

    def set_grl_alpha(self, alpha: float):

        self.grl.set_alpha(alpha)
        self.grl_alpha = alpha

    def get_feature_dim(self) -> int:

        return self.hidden


@gin.configurable("SimpleDANNTransformer")
class SimpleDANNTransformer(DANNTransformer):


    def __init__(
        self,
        input_size,
        hidden: int = 64,
        heads: int = 2,
        depth: int = 1,
        num_classes: int = 2,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden=hidden,
            heads=heads,
            depth=depth,
            num_classes=num_classes,
            discriminator_hidden_dims=[128], 
            **kwargs,
        )
        logging.info(f"Initialized Simple DANN Transformer with hidden={hidden}, depth={depth}")
