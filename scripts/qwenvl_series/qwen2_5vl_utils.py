import torch
from torch import nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, Cache, FlashAttentionKwargs, ALL_ATTENTION_FUNCTIONS, Unpack, Callable, \
    repeat_kv, TransformersKwargs
from types import MethodType
import math
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from scripts.verda import use_verda

    
def get_qwen_model(model_path, device, attn_implementation="eager"):
    model_kwargs = {
        "torch_dtype": "auto",
        "attn_implementation": attn_implementation,
        "low_cpu_mem_usage": True,
    }
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    
    if getattr(processor, "patch_size", None) is None:
        processor.patch_size = model.config.vision_config.patch_size

    conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "{question}"}
                ],
            },
        ]
    
    template = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    ans_start = "assistant\n"

    vision_start = model.config.vision_start_token_id
    image_start = ((processor(text=template, return_tensors="pt")['input_ids']) == vision_start).nonzero()[0, 1].item() + 1

    model.to(device)

    return processor, model, template, ans_start, image_start



def replace_qwen2_5vl_attention(image_start, image_end, rho, gamma):
    def using_verda_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    
        attn_weights = use_verda(
            attn_weights,
            image_start,
            image_end,
            rho=rho,
            min_keep=gamma
        )
        
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.config.rope_parameters["mrope_section"]
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, using_verda_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            position_ids=position_ids,  # pass positions for FA2
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    return forward


def replace_qwen2_5vl_forward(model, img_start, img_end, rho, gamma, target_layers):
    # Language model type: <class 'transformers.models.qwen.modeling_qwen.QwenForCausalLM'>
    print("Replacing Qwen Attention forward function in layers:", target_layers)
    # print("Model:", model)
    for idx, layer in enumerate(model.model.language_model.layers):
        if idx in target_layers:
            layer.self_attn.forward = MethodType(
                replace_qwen2_5vl_attention(img_start, img_end, rho=rho, gamma=gamma),
                layer.self_attn
            )  # replace Attention forward function
