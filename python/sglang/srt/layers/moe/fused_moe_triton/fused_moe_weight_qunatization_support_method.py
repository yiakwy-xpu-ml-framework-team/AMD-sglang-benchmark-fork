from abc import abstractmethod
from enum import Enum

import torch

from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"

class FusedMoEMethodBase(QuantizeMethodBase):

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
    ) -> torch.Tensor:
        raise NotImplementedError