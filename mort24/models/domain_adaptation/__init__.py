"""Domain Adaptation module for ICU benchmarks.

This module provides implementations of domain adaptation methods for improving
cross-dataset generalization in ICU prediction tasks.
"""

from mort24.models.domain_adaptation.gradient_reversal import (
    GradientReversalLayer,
    GradientReversalFunction,
    get_grl_alpha,
)
from mort24.models.domain_adaptation.domain_discriminator import (
    DomainDiscriminator,
    SimpleDomainDiscriminator,
)
from mort24.models.domain_adaptation.dann import DANNTransformer, SimpleDANNTransformer

__all__ = [
    "GradientReversalLayer",
    "GradientReversalFunction",
    "get_grl_alpha",
    "DomainDiscriminator",
    "SimpleDomainDiscriminator",
    "DANNTransformer",
    "SimpleDANNTransformer",
]
