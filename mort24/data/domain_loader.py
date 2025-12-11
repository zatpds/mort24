"""Multi-domain dataset loader for domain adaptation.

This module provides dataset classes that handle data from multiple domains
(e.g., eICU and MIMIC) for domain adaptation tasks.
"""

import logging
from typing import List, Optional, Union, Tuple
import gin
import numpy as np
import polars as pl
from torch import Tensor, from_numpy
from torch.utils.data import Dataset

from mort24.data.loader import PredictionPolarsDataset
from mort24.data.constants import DataSegment, DataSplit


@gin.configurable("MultiDomainDataset")
class MultiDomainDataset(Dataset):
    """Dataset for multi-domain learning with domain adaptation.

    This dataset combines source and target domain data, adding domain labels
    to enable adversarial domain adaptation training.

    Args:
        source_data: Dictionary containing source domain data (e.g., eICU)
        target_data: Dictionary containing target domain data (e.g., MIMIC)
        split: Data split to use (train/val/test)
        vars: Variable configuration dictionary
        sampling_strategy: How to sample from domains:
            - "balanced": Equal samples from source and target
            - "source_heavy": More samples from source (useful when target is unlabeled)
            - "target_heavy": More samples from target
            - "sequential": First source, then target
        source_target_ratio: Ratio of source to target samples (default: 1.0 for balanced)
        ram_cache: Whether to cache dataset in RAM
        name: Dataset name for logging
        grouping_segment: Segment to use for grouping (default: outcome)

    Returns:
        Tuple of (data, labels, mask, domain_id) where domain_id is 0 for source, 1 for target
    """

    def __init__(
        self,
        source_data: dict,
        target_data: dict,
        split: str = DataSplit.train,
        vars: dict = gin.REQUIRED,
        sampling_strategy: str = "balanced",
        source_target_ratio: float = 1.0,
        ram_cache: bool = True,
        name: str = "multi_domain",
        grouping_segment: str = DataSegment.outcome,
        mps: bool = False,
    ):
        """Initialize the multi-domain dataset."""
        super().__init__()

        self.split = split
        self.vars = vars
        self.sampling_strategy = sampling_strategy
        self.source_target_ratio = source_target_ratio
        self.name = name
        self.mps = mps

        # Create individual domain datasets
        self.source_dataset = PredictionPolarsDataset(
            data=source_data,
            split=split,
            vars=vars,
            grouping_segment=grouping_segment,
            ram_cache=ram_cache,
            mps=mps,
            name=f"{name}_source",
        )

        self.target_dataset = PredictionPolarsDataset(
            data=target_data,
            split=split,
            vars=vars,
            grouping_segment=grouping_segment,
            ram_cache=ram_cache,
            mps=mps,
            name=f"{name}_target",
        )

        self.source_len = len(self.source_dataset)
        self.target_len = len(self.target_dataset)

        # Calculate total length based on sampling strategy
        if sampling_strategy == "balanced":
            # Use the larger dataset size and cycle through the smaller one
            self.total_len = max(self.source_len, self.target_len) * 2
        elif sampling_strategy == "source_heavy":
            self.total_len = int(self.source_len * (1.0 + 1.0 / source_target_ratio))
        elif sampling_strategy == "target_heavy":
            self.total_len = int(self.target_len * (1.0 + source_target_ratio))
        elif sampling_strategy == "sequential":
            self.total_len = self.source_len + self.target_len
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

        # Calculate max sequence length across both domains
        self.maxlen = max(self.source_dataset.maxlen, self.target_dataset.maxlen)

        logging.info(
            f"Initialized MultiDomainDataset with {self.source_len} source and "
            f"{self.target_len} target samples. Total length: {self.total_len}"
        )

    def __len__(self) -> int:
        """Return total dataset length."""
        return self.total_len

    def _get_domain_sample_idx(self, idx: int) -> Tuple[int, int]:
        """Determine which domain to sample from and the index within that domain.

        Args:
            idx: Global index

        Returns:
            Tuple of (domain_id, domain_idx) where domain_id is 0 for source, 1 for target
        """
        if self.sampling_strategy == "balanced":
            # Alternate between source and target
            if idx % 2 == 0:
                domain_id = 0
                domain_idx = (idx // 2) % self.source_len
            else:
                domain_id = 1
                domain_idx = (idx // 2) % self.target_len

        elif self.sampling_strategy == "source_heavy":
            # More source samples
            threshold = int(self.total_len * self.source_target_ratio / (1.0 + self.source_target_ratio))
            if idx < threshold:
                domain_id = 0
                domain_idx = idx % self.source_len
            else:
                domain_id = 1
                domain_idx = (idx - threshold) % self.target_len

        elif self.sampling_strategy == "target_heavy":
            # More target samples
            threshold = int(self.total_len / (1.0 + self.source_target_ratio))
            if idx < threshold:
                domain_id = 0
                domain_idx = idx % self.source_len
            else:
                domain_id = 1
                domain_idx = (idx - threshold) % self.target_len

        elif self.sampling_strategy == "sequential":
            # First all source, then all target
            if idx < self.source_len:
                domain_id = 0
                domain_idx = idx
            else:
                domain_id = 1
                domain_idx = idx - self.source_len
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

        return domain_id, domain_idx

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get a sample from the multi-domain dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (data, labels, mask, domain_id)
                - data: Feature tensor (seq_len, num_features)
                - labels: Label tensor (seq_len,)
                - mask: Padding mask tensor (seq_len,)
                - domain_id: Domain identifier (0 for source, 1 for target)
        """
        # Determine which domain to sample from
        domain_id, domain_idx = self._get_domain_sample_idx(idx)

        # Get sample from appropriate domain
        if domain_id == 0:
            data, labels, mask = self.source_dataset[domain_idx]
        else:
            data, labels, mask = self.target_dataset[domain_idx]

        # Pad to max length if needed (to ensure consistent shapes across domains)
        if data.shape[0] < self.maxlen:
            import torch
            length_diff = self.maxlen - data.shape[0]
            pad_value = 0.0

            data_pad = torch.ones((length_diff, data.shape[1])) * pad_value
            data = torch.cat([data, data_pad], dim=0)

            labels_pad = torch.ones(length_diff) * pad_value
            labels = torch.cat([labels, labels_pad], dim=0)

            mask_pad = torch.zeros(length_diff, dtype=torch.bool)
            mask = torch.cat([mask, mask_pad], dim=0)

        # Convert domain_id to tensor
        domain_id_tensor = from_numpy(np.array(domain_id, dtype=np.int64))

        return data, labels, mask, domain_id_tensor

    def get_balance(self) -> list:
        """Return the weight balance across both domains.

        Returns:
            Weights for each label class
        """
        # Combine balance from both domains
        source_balance = self.source_dataset.get_balance()
        target_balance = self.target_dataset.get_balance()

        # Average the balances
        combined_balance = [(s + t) / 2.0 for s, t in zip(source_balance, target_balance)]
        return combined_balance

    def get_source_dataset(self) -> PredictionPolarsDataset:
        """Get the source domain dataset.

        Returns:
            Source domain dataset
        """
        return self.source_dataset

    def get_target_dataset(self) -> PredictionPolarsDataset:
        """Get the target domain dataset.

        Returns:
            Target domain dataset
        """
        return self.target_dataset

    def get_feature_names(self) -> List[str]:
        """Get feature names (assumes both domains have same features).

        Returns:
            List of feature names
        """
        return self.source_dataset.get_feature_names()


@gin.configurable("SourceOnlyDataset")
class SourceOnlyDataset(PredictionPolarsDataset):
    """Wrapper for source-only baseline (no domain adaptation).

    This is a simple wrapper that adds a domain_id tensor to the output
    for compatibility with domain adaptation code, but always uses source domain.

    Args:
        Same as PredictionPolarsDataset
    """

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get a sample with domain_id added.

        Args:
            idx: Sample index

        Returns:
            Tuple of (data, labels, mask, domain_id) where domain_id is always 0
        """
        data, labels, mask = super().__getitem__(idx)
        domain_id = from_numpy(np.array(0, dtype=np.int64))  # Always source
        return data, labels, mask, domain_id
