import math
import torch


def use_verda(
    attn_weights,
    img_start,
    img_end,
    rho,  # Visual attention coverage ratio.
    min_keep,  # Minimum retained visual-token ratio (gamma).
    eps=1e-6,
):
    # Visual Cues Reinforcement via Head-Adaptive De-redundancy and Distance-Aware Attenuation.
    lambda_decay = 0.5  
    device = attn_weights.device
    attn_pruned = attn_weights.clone()

    # -------------------------
    # 1. Head-Adaptive Visual De-redundancy
    # -------------------------
    img_attn = attn_pruned[:, :, -1, img_start:img_end]   # [bsz, heads, n_img]
    bsz, n_heads, n_img = img_attn.shape

    img_attn_valid = img_attn.masked_fill(torch.isinf(img_attn), 0.0).clamp(min=0.0)

    total_head_attn = img_attn_valid.sum(dim=-1, keepdim=True)   # [bsz, heads, 1]

    score = img_attn_valid / (total_head_attn + eps)             # [bsz, heads, n_img]

    sorted_score, sorted_idx = torch.sort(score, dim=-1, descending=True)   # [bsz, heads, n_img]
    sorted_attn = torch.gather(img_attn_valid, dim=-1, index=sorted_idx)    # [bsz, heads, n_img]

    cumsum_attn = torch.cumsum(sorted_attn, dim=-1)                         # [bsz, heads, n_img]

    threshold = rho * total_head_attn                                       # [bsz, heads, 1]

    k_min = max(1, math.ceil(0.1 * n_img))

    cover_mask = cumsum_attn >= threshold                                   # [bsz, heads, n_img]
    first_true_idx = cover_mask.float().argmax(dim=-1)                      # [bsz, heads]
    has_valid = cover_mask.any(dim=-1)
    first_true_idx = torch.where(
        has_valid,
        first_true_idx,
        torch.full_like(first_true_idx, n_img - 1)
    )

    k_per_head = torch.clamp(first_true_idx + 1, min=k_min, max=n_img)      # [bsz, heads]
    rank = torch.arange(n_img, device=device).view(1, 1, n_img)             # [1,1,n_img]
    keep_sorted_mask = rank < k_per_head.unsqueeze(-1)                      # [bsz, heads, n_img]

    mask_img = torch.zeros_like(img_attn, dtype=torch.bool)                 # [bsz, heads, n_img]
    mask_img.scatter_(-1, sorted_idx, keep_sorted_mask)


    img_removed = img_attn.masked_fill(mask_img, 0.0)                       
    img_attn_kept = img_attn.masked_fill(~mask_img, 0.0)                    

    attn_pruned[:, :, -1, img_start:img_end] = img_attn_kept

    # -------------------------
    # 2. Distance-Aware Attenuation
    # -------------------------
    text_end = attn_weights.size(-1)
    prefix_attn = attn_pruned[:, :, -1, :img_start]
    suffix_attn = attn_pruned[:, :, -1, img_end:text_end]

    text_attn = torch.cat([prefix_attn, suffix_attn], dim=-1)               # [bsz, heads, n_text]
    num_text_tokens = text_attn.shape[-1]

    token_idx = torch.arange(num_text_tokens, device=device, dtype=text_attn.dtype)
    dist = torch.abs(token_idx - (num_text_tokens - 1))                     # [n_text]
    dist_norm = dist / (dist.max() + eps)
    dist_norm = dist_norm * (1.0 - lambda_decay) + lambda_decay         # [n_text]
    dist_norm = dist_norm.view(1, 1, num_text_tokens)                       # [1,1,n_text]

    attn_sum = text_attn.masked_fill(torch.isinf(text_attn), 0.0).sum(dim=1)  # [bsz, n_text]
    attn_norm = torch.sigmoid(attn_sum).unsqueeze(1)                           # [bsz, 1, n_text]
    
    img_attn_for_head = attn_pruned[:, :, -1, img_start:img_end]               # [bsz, heads, n_img]
    img_attn_valid2 = img_attn_for_head.masked_fill(torch.isinf(img_attn_for_head), 0.0).clamp(min=0.0)
    head_visual_score = img_attn_valid2.sum(dim=-1, keepdim=True)              # [bsz, heads, 1]
    head_gate = (1.0 - head_visual_score).clamp(min=0.0)                       # [bsz, heads, 1]

    decay_factor = (dist_norm * attn_norm * head_gate).to(text_attn.dtype)     # [bsz, heads, n_text]
    text_attn_decayed = text_attn * decay_factor

    text_removed = (text_attn - text_attn_decayed).clamp(min=0.0)              # [bsz, heads, n_text]

    n_prefix = prefix_attn.shape[-1]
    prefix_decayed = text_attn_decayed[:, :, :n_prefix]
    suffix_decayed = text_attn_decayed[:, :, n_prefix:]

    attn_pruned[:, :, -1, :img_start] = prefix_decayed
    attn_pruned[:, :, -1, img_end:text_end] = suffix_decayed

    # -------------------------
    # 3. Visual Cues Head-Adaptive Enhancement
    # -------------------------
    distribute = (
        img_removed.sum(dim=-1, keepdim=True) +
        text_removed.sum(dim=-1, keepdim=True)
    )   # [bsz, heads, 1]

    k_per_head_float = k_per_head.unsqueeze(-1).to(distribute.dtype).clamp_min(1.0)  # [bsz, heads, 1]
    distribute_per_token = distribute / k_per_head_float                               # [bsz, heads, 1]

    distribute_map = distribute_per_token.expand(-1, -1, n_img) * mask_img.to(distribute.dtype)

    attn_slice = attn_pruned[:, :, -1, img_start:img_end]   # [bsz, heads, n_img]
    attn_slice = attn_slice + distribute_map
    attn_pruned[:, :, -1, img_start:img_end] = attn_slice

    return attn_pruned