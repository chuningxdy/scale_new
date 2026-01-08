"""
Pure-PyTorch StripedHyena-style hybrid (NO CUDA extensions required)

Goal: a practical "StripedHyena-like" model that runs at ~2k context without
flash-attn / mamba-ssm / custom kernels.

Key changes vs your script:
- Replace slow Python-loop selective scan with a Hyena-like long depthwise Conv1d + gating
- Use PyTorch SDPA for attention (fastest available built-in; no flash-attn install needed)
- Proper decoder block structure: RMSNorm -> sublayer -> residual, repeated for MLP
- Optional RoPE + GQA
- Flexible striping schedule (alternate / every-4 / every-8 / ssm_only / attn_only)
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


# ----------------------------
# Config
# ----------------------------

class StripedHyenaLiteConfig(PretrainedConfig):
    model_type = "striped_hyena_lite"

    def __init__(
        self,
        vocab_size: int = 3000,
        d_model: int = 512,
        n_layers: int = 12,
        # Attention
        n_heads: int = 8,
        n_kv_heads: int | None = None,      # GQA if < n_heads
        head_dim: int = 64,                  # per-head dim; attention projection dim = n_heads * head_dim
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048, # you said 2k context is enough
        # Hyena-like mixing
        d_inner: int = 2048,                 # "glu_size" / MLP inner
        hyena_kernel_size: int = 127,        # long depthwise conv kernel; try 63/127/255 for 2k
        # Striping
        hybrid_pattern: str = "every_4",     # "alternate", "every_4", "every_8", "ssm_only", "attn_only"
        # Misc
        dropout: float = 0.0,
        bias: bool = False,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_groups = self.n_heads // self.n_kv_heads
        self.head_dim = head_dim

        # Sanity: attention projection dim should usually match d_model, but doesn't have to.
        # Many models use head_dim = d_model / n_heads.
        # If you want that behavior, set head_dim = d_model // n_heads.
        self.attn_proj_dim = self.n_heads * self.head_dim

        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.d_inner = d_inner
        self.hyena_kernel_size = hyena_kernel_size

        self.hybrid_pattern = hybrid_pattern
        self.dropout = dropout
        self.bias = bias

        self.attn_layer_idxs, self.ssm_layer_idxs = self._get_layer_pattern()

    def _get_layer_pattern(self):
        L = self.n_layers
        pat = self.hybrid_pattern
        if pat == "alternate":
            # even=hyena/ssm-like, odd=attention
            ssm = list(range(0, L, 2))
            attn = list(range(1, L, 2))
        elif pat == "every_4":
            # attention every 4 layers, like [3,7,11,...]
            attn = list(range(3, L, 4))
            ssm = [i for i in range(L) if i not in attn]
        elif pat == "every_8":
            attn = list(range(7, L, 8))
            ssm = [i for i in range(L) if i not in attn]
        elif pat == "ssm_only":
            attn, ssm = [], list(range(L))
        elif pat == "attn_only":
            attn, ssm = list(range(L)), []
        else:
            # default
            attn = list(range(3, L, 4))
            ssm = [i for i in range(L) if i not in attn]
        return attn, ssm


# ----------------------------
# Norm
# ----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ----------------------------
# RoPE (for attention)
# ----------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE requires even head_dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: (B, H, L, D); cos,sin: (L, D)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,L,D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


# ----------------------------
# Attention (dense, but uses PyTorch SDPA)
# ----------------------------

class SDPAWithRoPEGQA(nn.Module):
    """
    Dense causal attention using torch.scaled_dot_product_attention (SDPA).
    No flash-attn install required; PyTorch will pick the best available kernel.
    """
    def __init__(self, config: StripedHyenaLiteConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.proj_dim = config.attn_proj_dim

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.proj_dim, config.d_model, bias=config.bias)

        self.rope = RotaryEmbedding(self.head_dim, config.max_position_embeddings, base=config.rope_theta)
        self.dropout_p = float(config.dropout)

    def forward(self, x):
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)      # (B,H,L,D)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,Hkv,L,D)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,Hkv,L,D)

        cos, sin = self.rope(x, L)
        # Apply RoPE to q and k (k has Hkv heads, still fine)
        q, k = apply_rope(q, k, cos, sin)

        # GQA: expand k,v along head dimension to match q heads
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)  # (B,H,L,D)
            v = v.repeat_interleave(self.n_groups, dim=1)  # (B,H,L,D)

        # SDPA causal attention
        # dropout_p only applied during training
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )  # (B,H,L,D)

        out = out.transpose(1, 2).contiguous().view(B, L, self.proj_dim)
        return self.o_proj(out)


# ----------------------------
# Hyena-like long mixing (pure PyTorch)
# ----------------------------

class HyenaLikeMix(nn.Module):
    """
    Pure PyTorch "Hyena-ish" long mixing:
      x -> in_proj -> (value, gate)
      value -> depthwise causal Conv1d (long kernel)
      gated -> out_proj
    """
    def __init__(self, config: StripedHyenaLiteConfig):
        super().__init__()
        d = config.d_model
        m = config.d_inner
        k = config.hyena_kernel_size

        self.in_proj = nn.Linear(d, 2 * m, bias=config.bias)

        # Depthwise conv (groups=m) => O(L*m*k), fine for L~2k and moderate k (63/127/255)
        self.dwconv = nn.Conv1d(
            in_channels=m,
            out_channels=m,
            kernel_size=k,
            padding=k - 1,     # causal-ish; we trim back to length L
            groups=m,
            bias=True
        )

        self.out_proj = nn.Linear(m, d, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        B, L, _ = x.shape
        v, g = self.in_proj(x).chunk(2, dim=-1)  # (B,L,m) each

        v = v.transpose(1, 2)                    # (B,m,L)
        v = self.dwconv(v)[..., :L]              # trim to causal length
        v = v.transpose(1, 2)                    # (B,L,m)

        v = F.silu(v)
        y = v * F.silu(g)
        return self.drop(self.out_proj(y))


class SwiGLUMLP(nn.Module):
    def __init__(self, config: StripedHyenaLiteConfig):
        super().__init__()
        d = config.d_model
        m = config.d_inner
        self.gate = nn.Linear(d, m, bias=config.bias)
        self.up   = nn.Linear(d, m, bias=config.bias)
        self.down = nn.Linear(m, d, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ----------------------------
# Blocks
# ----------------------------

class AttentionBlock(nn.Module):
    def __init__(self, config: StripedHyenaLiteConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = SDPAWithRoPEGQA(config)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HyenaBlock(nn.Module):
    def __init__(self, config: StripedHyenaLiteConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.mix = HyenaLikeMix(config)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x):
        x = x + self.mix(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HybridBlock(nn.Module):
    def __init__(self, config: StripedHyenaLiteConfig, layer_idx: int):
        super().__init__()
        if layer_idx in config.attn_layer_idxs:
            self.block = AttentionBlock(config)
            self.block_type = "attention"
        else:
            self.block = HyenaBlock(config)
            self.block_type = "hyena"

    def forward(self, x):
        return self.block(x)


# ----------------------------
# Model
# ----------------------------

class StripedHyenaLiteForCausalLM(PreTrainedModel):
    config_class = StripedHyenaLiteConfig

    def __init__(self, config: StripedHyenaLiteConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([HybridBlock(config, i) for i in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.post_init()

        # Helpful printout
        print("\nStripedHyenaLite (no CUDA extensions) created:")
        print(f"  d_model={config.d_model}, n_layers={config.n_layers}, d_inner={config.d_inner}")
        print(f"  attn: n_heads={config.n_heads}, n_kv_heads={config.n_kv_heads}, head_dim={config.head_dim}")
        print(f"  rope: theta={config.rope_theta}, max_pos={config.max_position_embeddings}")
        print(f"  hyena_kernel_size={config.hyena_kernel_size}")
        print(f"  pattern={config.hybrid_pattern}")
        print(f"  attn_layers={config.attn_layer_idxs}")
        print(f"  hyena_layers={config.ssm_layer_idxs}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embedding(input_ids)  # (B,L,d)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_params_by_block_type(self):
        attn_params = 0
        hyena_params = 0
        other_params = 0

        # Embeddings + head + final norm counted as "other"
        other_params += sum(p.numel() for p in self.embedding.parameters())
        other_params += sum(p.numel() for p in self.norm_f.parameters())
        other_params += sum(p.numel() for p in self.lm_head.parameters())

        for blk in self.layers:
            n = sum(p.numel() for p in blk.parameters())
            if blk.block_type == "attention":
                attn_params += n
            else:
                hyena_params += n

        return {"attn": attn_params, "hyena": hyena_params, "other": other_params}


# ----------------------------
# Quick sanity test
# ----------------------------
if __name__ == "__main__":
    cfg = StripedHyenaLiteConfig(
        vocab_size=3000,
        d_model=512,
        n_layers=12,
        n_heads=8,
        n_kv_heads=2,          # e.g., GQA
        head_dim=64,           # if you want d_model match: set head_dim = d_model//n_heads (=64 here)
        d_inner=2048,
        hyena_kernel_size=127,
        hybrid_pattern="every_4",
        max_position_embeddings=2048,
        dropout=0.0,
        bias=False,
    )

    model = StripedHyenaLiteForCausalLM(cfg)
    print("Total params:", model.count_parameters())
    print("Params by type:", model.count_params_by_block_type())

    B, L = 2, 128
    input_ids = torch.randint(0, cfg.vocab_size, (B, L))
    out = model(input_ids)
    print("logits:", out.logits.shape)
