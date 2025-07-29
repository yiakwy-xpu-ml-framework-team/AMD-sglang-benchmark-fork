from argparse import ArgumentParser
from typing import Optional, Tuple

import torch
import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache

# Monkey patch to Qwen2.5 modeling
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.processing_utils import Unpack


class RotatedQwenDecoderLayer(Qwen2DecoderLayer):

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        hidden_size = config.hidden_size

        self.R1 = torch.nn.Parameter(torch.eye(hidden_size, dtype=config.torch_dtype))

        # for debug prupose
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        # rotate hidden states
        hidden_states = torch.matmul(hidden_states, self.R1.T)

        # Original Qwen forward routine
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention, note self_attn_weights will be used later
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # rotate back hiddenstates
        hidden_states = torch.matmul(hidden_states, self.R1)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


OldQwen2DecoderLayer = Qwen2DecoderLayer


def convert_pth_to_safetensors(pth_path, hf_tokenizer_path, config_path, output_dir):
    config = AutoConfig.from_pretrained(config_path)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    for layer_idx in range(config.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        print(f"[Layer#{layer_idx}] {layer}")

    print("initialize model weights ...")

    state_dict = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False, assign=True)

    print("verify the model ...")

    verify_tokenizer_and_model(hf_tokenizer_path, model)

    print("save model ...")

    model.save_pretrained(output_dir, safe_serialization=True)


def verify_tokenizer_and_model(hf_tokenizer_path, model):
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_path)

    texts = [
        "介绍一下大熊猫"
    ]  # ["Give me a short introduction to large language model.", ]
    messages = [{"role": "user", "content": text} for text in texts]

    prompts = hf_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = hf_tokenizer([prompts], return_tensors="pt").to(model.device)
    outputs_ids = model.generate(**model_inputs, max_new_tokens=16)

    outputs_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, outputs_ids)
    ]

    response = hf_tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)[0]
    print(f"response : {response}")


def load_and_verify_hf_model(source_model):
    model = AutoModelForCausalLM.from_pretrained(
        source_model, torch_dtype="auto", device_map="auto"
    )

    verify_tokenizer_and_model(source_model, model)
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source_model", default=None, type=str, required=False, help="source model."
    )
    parser.add_argument(
        "--checkpoint_file",
        default=None,
        type=str,
        required=False,
        help="Name of the checkpoint file in the repo.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="Where to save the converted model.",
    )
    parser.add_argument(
        "--hf_tokenizer_path",
        default=None,
        type=str,
        required=False,
        help="Path to the tokenizer file to use.",
    )
    args = parser.parse_args()

    if args.source_model:
        load_and_verify_hf_model(args.source_model)
    else:
        transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer = (
            RotatedQwenDecoderLayer
        )

        convert_pth_to_safetensors(
            args.checkpoint_file,
            args.hf_tokenizer_path,
            args.output_dir,
            args.output_dir,
        )
