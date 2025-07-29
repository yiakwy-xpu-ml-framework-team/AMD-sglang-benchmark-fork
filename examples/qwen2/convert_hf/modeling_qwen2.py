import transformers
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config as _Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM as _Qwen2ForCausalLM,
)

from .convert_zero_pruned_checkpoint_to_hf import RotatedQwenDecoderLayer

transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer = RotatedQwenDecoderLayer

import logging

_logger = logging.getLogger()


class Qwen2Config(_Qwen2Config):
    pass


_logger.info("Using Monkey Patched Qwen2 (RotatedQwenDecoderLayer) model")


# monkey patched model
class Qwen2ForCausalLM(_Qwen2ForCausalLM):
    pass
