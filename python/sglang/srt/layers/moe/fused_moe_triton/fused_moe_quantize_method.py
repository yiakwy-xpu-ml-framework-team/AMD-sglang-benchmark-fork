from abc import abstractmethod
from enum import Enum
from typing import Callable, List, Optional, Tuple

import torch

from sglang.srt.custom_op import CustomOp

from sglang.srt.layers.moe.fused_moe_native import moe_forward_native
from sglang.srt.layers.moe.topk import select_experts

from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.srt.layers.quantization.int8_kernel import (
    per_token_group_quant_int8,
    per_token_quant_int8,
)

from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.utils import get_bool_env_var, is_hip, permute_weight, set_weight_attrs

if torch.cuda.is_available():
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
else:
    fused_experts = None  # type: ignore

from sglang.srt.utils import (
    direct_register_custom_op,
    get_bool_env_var,
    get_device_name,
    is_cuda,
    is_hip,
)

import logging

logger = logging.getLogger(__name__)

_is_hip = is_hip()

if _is_hip:
    from aiter import ck_moe

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

# class FusedMoeWeightScaleSupported(Enum):
#     TENSOR = "tensor"
#     CHANNEL = "channel"
#     GROUP = "group"
#     BLOCK = "block"

from sglang.srt.layers.moe.fused_moe_triton.fused_moe_weight_qunatization_support_method import FusedMoeWeightScaleSupported

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


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_hip and get_bool_env_var("CK_MOE"):
            layer.w13_weight = torch.nn.Parameter(
                permute_weight(layer.w13_weight.data),
                requires_grad=False,
            )
            torch.cuda.empty_cache()
            layer.w2_weight = torch.nn.Parameter(
                permute_weight(layer.w2_weight.data),
                requires_grad=False,
            )
            torch.cuda.empty_cache()
        return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        inplace: bool = True,
        no_combine: bool = False,
    ) -> torch.Tensor:
        return self.forward(
            x=x,
            layer=layer,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            activation=activation,
            inplace=inplace,
            no_combine=no_combine,
        )

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        inplace: bool = True,
        no_combine: bool = False,
    ) -> torch.Tensor:
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
        )

        if _is_hip and get_bool_env_var("CK_MOE"):
            assert not no_combine, "unsupported"
            return ck_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                None,
                None,
                None,
                None,
                32,
                None,
                activation,
            )
        else:
            return fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=inplace and not no_combine,
                activation=activation,
                no_combine=no_combine,
            )

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        inplace: bool = True,
    ) -> torch.Tensor:
        return moe_forward_native(
            layer,
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
            custom_routing_function,
            correction_bias,
        )

    def forward_tpu(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("The TPU backend currently does not support MoE.")

    forward_native = forward_cuda