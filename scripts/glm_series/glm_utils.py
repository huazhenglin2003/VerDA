import math
from types import MethodType
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.verda import use_verda


def get_glm_model(model_path, device, attn_implementation="eager"):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=attn_implementation, 
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    if hasattr(model.config, "vision_config"):
        vision_cfg = model.config.vision_config
        if isinstance(vision_cfg, dict):
            image_size = vision_cfg["image_size"]
            patch_size = vision_cfg["patch_size"]
        else:
            image_size = vision_cfg.image_size
            patch_size = vision_cfg.patch_size

        num_image_tokens = (image_size // patch_size // 2) ** 2
    else:
        num_image_tokens = None

    model.to(device)
    model.eval()
    return tokenizer, model, num_image_tokens


def build_core_attention_forward(image_start, image_end, rho, gamma):
    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        if not hasattr(self, "_verda_debug_counter"):
            self._verda_debug_counter = 0
        self._verda_debug_counter += 1
        if self._verda_debug_counter <= 5:
            print(
                f"[VerDA HIT] core_attention.forward called "
                f"{self._verda_debug_counter} times | "
                f"q={tuple(query_layer.shape)}, k={tuple(key_layer.shape)}, v={tuple(value_layer.shape)}"
            )


        original_layout_seq_first = False

        if query_layer.dim() != 4:
            raise ValueError(f"Unexpected query_layer shape: {query_layer.shape}")

        if query_layer.shape[0] > 1 and query_layer.shape[1] <= 16:
            # [sq, b, np, hn] -> [b, np, sq, hn]
            query_layer = query_layer.permute(1, 2, 0, 3).contiguous()
            key_layer = key_layer.permute(1, 2, 0, 3).contiguous()
            value_layer = value_layer.permute(1, 2, 0, 3).contiguous()
            original_layout_seq_first = True
        else:
            # assume already [b, np, sq, hn]
            pass

        bsz, n_heads, q_len, head_dim = query_layer.shape
        _, _, k_len, _ = key_layer.shape

        # [b, np, sq, sk]
        attention_scores = torch.matmul(
            query_layer,
            key_layer.transpose(-1, -2)
        )

        norm_factor = getattr(self, "norm_factor", math.sqrt(head_dim))
        attention_scores = attention_scores / norm_factor

        coeff = getattr(self, "coeff", None)
        if coeff is not None:
            attention_scores = attention_scores * coeff

        if attention_mask is None and q_len == k_len:
            causal_mask = torch.ones(
                (1, 1, q_len, k_len),
                dtype=torch.bool,
                device=attention_scores.device
            ).triu(diagonal=1)
            attention_scores = attention_scores.masked_fill(causal_mask, float("-inf"))
        elif attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                if attention_mask.shape[-1] == k_len:
                    mask_to_apply = attention_mask
                    if mask_to_apply.ndim == 4:
                        test_val = mask_to_apply[..., -1, -1].float().mean().item()
                        if test_val > 0.5:
                            mask_to_apply = ~mask_to_apply
                    else:
                        mask_to_apply = ~mask_to_apply
                    attention_scores = attention_scores.masked_fill(mask_to_apply, float("-inf"))
                else:
                    raise ValueError(f"Unexpected attention_mask shape: {attention_mask.shape}")
            else:
                # additive mask
                attention_scores = attention_scores + attention_mask

        if getattr(self, "attention_softmax_in_fp32", False):
            attention_probs = F.softmax(attention_scores.float(), dim=-1).to(value_layer.dtype)
        else:
            attention_probs = F.softmax(attention_scores, dim=-1)


        attention_probs = use_verda(
            attention_probs,
            img_start=image_start,
            img_end=image_end,
            rho=rho,
            min_keep=gamma
        )

        # dropout
        if hasattr(self, "attention_dropout") and self.attention_dropout is not None:
            attention_probs = self.attention_dropout(attention_probs)

        # [b, np, sq, hn]
        context_layer = torch.matmul(attention_probs, value_layer)

        if original_layout_seq_first:
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()  # [sq, b, np, hn]
            new_context_layer_shape = context_layer.size()[:-2] + (
                context_layer.size(-2) * context_layer.size(-1),
            )
            context_layer = context_layer.reshape(*new_context_layer_shape)
        else:
            context_layer = context_layer.transpose(1, 2).contiguous()  # [b, sq, np, hn]
            new_context_layer_shape = context_layer.size()[:-2] + (
                context_layer.size(-2) * context_layer.size(-1),
            )
            context_layer = context_layer.reshape(*new_context_layer_shape)

        return context_layer

    return forward


def replace_glm_forward(
    model,
    image_start,
    image_end,
    rho,
    gamma,
    target_layers=None,
):
    if not hasattr(model, "transformer"):
        raise AttributeError("model has no attribute 'transformer'")
    if not hasattr(model.transformer, "encoder"):
        raise AttributeError("model.transformer has no attribute 'encoder'")
    if not hasattr(model.transformer.encoder, "layers"):
        raise AttributeError("model.transformer.encoder has no attribute 'layers'")

    layers = model.transformer.encoder.layers

    if target_layers is None:
        target_layers = list(range(len(layers)))

    print(f"[Patch] Replacing GLM core_attention.forward in layers: {target_layers}")

    replaced = []
    for layer_idx in target_layers:
        block = layers[layer_idx]

        if not hasattr(block, "self_attention"):
            print(f"[Skip] Layer {layer_idx} has no self_attention")
            continue
        if not hasattr(block.self_attention, "core_attention"):
            print(f"[Skip] Layer {layer_idx} has no core_attention")
            continue

        core_attn = block.self_attention.core_attention
        new_forward = build_core_attention_forward(
            image_start=image_start,
            image_end=image_end,
            rho=rho,
            gamma=gamma,
        )

        core_attn.forward = MethodType(new_forward, core_attn)
        core_attn._verda_patched = True
        replaced.append(layer_idx)
        print(f"[Done] Layer {layer_idx} replaced.")

    if len(replaced) == 0:
        raise RuntimeError("No layer was successfully patched.")

    print(f"[Patch] Successfully replaced layers: {replaced}")
