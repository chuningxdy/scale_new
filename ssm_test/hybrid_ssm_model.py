"""
Hybrid SSM/Attention Model (StripedHyena-style)
Combines SSM (Mamba) blocks with Attention blocks for efficient sequence modeling

Follows standard practices:
- RoPE (Rotary Position Embeddings) for attention
- GQA (Grouped Query Attention) option
- RMSNorm instead of LayerNorm
- Proper block structure with separate attention and MLP sub-layers
"""

import torch
import torch.nn as nn
import math
from transformers import PreTrainedModel, PretrainedConfig


class HybridSSMConfig(PretrainedConfig):
    """Configuration for Hybrid SSM/Attention model"""
    model_type = "hybrid_ssm"

    def __init__(
        self,
        vocab_size: int = 3000,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        n_kv_heads: int = None,  # For GQA: if None, use n_heads (standard MHA)
        glu_size: int = 2048,    # inner_mlp_size from table
        kv_size: int = 64,       # per-head dimension from table
        d_state: int = 16,       # SSM state dimension
        d_conv: int = 4,         # Convolution kernel size for SSM
        dropout: float = 0.0,
        bias: bool = False,
        hybrid_pattern: str = "alternate",  # "alternate", "ssm_heavy", or custom
        tie_word_embeddings: bool = True,
        rope_theta: float = 10000.0,  # RoPE base frequency
        max_position_embeddings: int = 131072,  # Max sequence length for RoPE
        **kwargs
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads  # GQA
        self.glu_size = glu_size
        self.kv_size = kv_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.dropout = dropout
        self.bias = bias
        self.hybrid_pattern = hybrid_pattern
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # Validate GQA configuration
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_groups = self.n_heads // self.n_kv_heads  # Number of query groups per KV head

        # Calculate which layers are attention vs SSM
        self.attn_layer_idxs, self.ssm_layer_idxs = self._get_layer_pattern()

    def _get_layer_pattern(self):
        """
        Determine which layers are attention vs SSM.

        Patterns:
        - "alternate": 50/50 split (StripedHyena-Hessian-7B style)
          SSM at even indices, Attention at odd indices
        - "ssm_heavy": ~75% SSM, ~25% attention (every 4th layer)
          Attention at [3, 7, 11, ...], SSM elsewhere
        - "evo2_7b": Evo2-7B style - attention every 4 layers
          Attention at [3, 7, 11, 15, ...], SSM elsewhere
        - "evo2_40b": Evo2-40B style - attention every 8 layers
          Attention at [7, 15, 23, 31, ...], SSM elsewhere
        """
        if self.hybrid_pattern == "alternate":
            # StripedHyena-Hessian: 50/50 alternating
            ssm_idxs = list(range(0, self.n_layers, 2))
            attn_idxs = list(range(1, self.n_layers, 2))
        elif self.hybrid_pattern == "ssm_heavy" or self.hybrid_pattern == "evo2_7b":
            # Evo2-7B: ~25% attention (every 4th layer, starting at 3)
            attn_idxs = list(range(3, self.n_layers, 4))
            ssm_idxs = [i for i in range(self.n_layers) if i not in attn_idxs]
        elif self.hybrid_pattern == "evo2_40b":
            # Evo2-40B: ~12.5% attention (every 8th layer, starting at 7)
            attn_idxs = list(range(7, self.n_layers, 8))
            ssm_idxs = [i for i in range(self.n_layers) if i not in attn_idxs]
        elif self.hybrid_pattern == "ssm_only":
            # Pure SSM (no attention)
            ssm_idxs = list(range(self.n_layers))
            attn_idxs = []
        elif self.hybrid_pattern == "attn_only":
            # Pure attention (no SSM) - equivalent to standard transformer
            attn_idxs = list(range(self.n_layers))
            ssm_idxs = []
        else:
            # Default to alternate
            ssm_idxs = list(range(0, self.n_layers, 2))
            attn_idxs = list(range(1, self.n_layers, 2))
        return attn_idxs, ssm_idxs


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used by LLaMA, StripedHyena)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 131072, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype)
        )


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to query and key tensors."""
    # cos, sin: (seq_len, head_dim)
    # q, k: (batch, n_heads, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SSMBlock(nn.Module):
    """Mamba-style SSM block"""
    def __init__(self, config: HybridSSMConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        d_state = config.d_state
        d_conv = config.d_conv
        d_inner = config.glu_size

        # Input projection (to 2x for gating)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=config.bias)

        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            bias=True
        )

        # SSM parameters
        dt_rank = math.ceil(d_model / 16)
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize A (state transition matrix)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)

        x = nn.functional.silu(x)
        x_ssm = self.ssm(x)
        x = x_ssm * nn.functional.silu(z)

        x = self.out_proj(x)
        x = self.dropout(x)
        return x

    def ssm(self, x):
        dt_b_c = self.x_proj(x)
        dt_rank = math.ceil(self.config.d_model / 16)
        dt, B, C = torch.split(
            dt_b_c,
            [dt_rank, self.config.d_state, self.config.d_state],
            dim=-1
        )
        dt = self.dt_proj(dt)
        dt = nn.functional.softplus(dt)
        A = -torch.exp(self.A_log)
        y = self.selective_scan(x, dt, A, B, C, self.D)
        return y

    def selective_scan(self, x, dt, A, B, C, D):
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1)
            A_discrete = torch.exp(A.unsqueeze(0) * dt_t)
            B_t = B[:, t, :].unsqueeze(1)
            B_discrete = B_t * dt_t

            h = h * A_discrete + x[:, t, :].unsqueeze(-1) * B_discrete

            C_t = C[:, t, :].unsqueeze(1)
            y_t = (h * C_t).sum(dim=-1) + x[:, t, :] * D
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y


class Attention(nn.Module):
    """Multi-head attention with RoPE and optional GQA"""
    def __init__(self, config: HybridSSMConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_groups = config.n_groups  # For GQA
        self.head_dim = config.kv_size
        self.d_model = config.d_model

        # Q projection: full heads
        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=config.bias)
        # K, V projections: potentially fewer heads for GQA
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=config.bias)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: expand K, V to match Q heads
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # Scaled dot-product attention with causal mask
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(batch, seq_len, self.n_heads * self.head_dim)
        out = self.o_proj(out)
        return out


class MLP(nn.Module):
    """Feed-forward MLP with SwiGLU activation (like LLaMA)"""
    def __init__(self, config: HybridSSMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.glu_size, bias=config.bias)
        self.up_proj = nn.Linear(config.d_model, config.glu_size, bias=config.bias)
        self.down_proj = nn.Linear(config.glu_size, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU: silu(gate) * up
        return self.dropout(self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)))


class AttentionBlock(nn.Module):
    """Attention block with proper structure: norm -> attn -> residual -> norm -> mlp -> residual"""
    def __init__(self, config: HybridSSMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.attn_norm(x))
        # MLP with residual
        x = x + self.mlp(self.mlp_norm(x))
        return x


class SSMLayerBlock(nn.Module):
    """SSM block with proper structure: norm -> ssm -> residual"""
    def __init__(self, config: HybridSSMConfig):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.ssm = SSMBlock(config)

    def forward(self, x):
        return x + self.ssm(self.norm(x))


class HybridBlock(nn.Module):
    """Hybrid block: either attention or SSM based on layer index"""
    def __init__(self, config: HybridSSMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        if layer_idx in config.attn_layer_idxs:
            self.block = AttentionBlock(config)
            self.block_type = "attention"
        else:
            self.block = SSMLayerBlock(config)
            self.block_type = "ssm"

    def forward(self, x):
        return self.block(x)


class HybridSSMModel(PreTrainedModel):
    """Hybrid SSM/Attention model for causal language modeling"""
    config_class = HybridSSMConfig

    def __init__(self, config: HybridSSMConfig):
        super().__init__(config)
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            HybridBlock(config, layer_idx) for layer_idx in range(config.n_layers)
        ])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Tie weights (handled by HuggingFace's post_init)
        self.post_init()

        # Print architecture summary
        print(f"\nHybrid SSM/Attention Model Architecture:")
        print(f"  d_model: {config.d_model}, n_layers: {config.n_layers}")
        print(f"  n_heads: {config.n_heads}, n_kv_heads: {config.n_kv_heads} (GQA groups: {config.n_groups})")
        print(f"  Attention layers: {len(config.attn_layer_idxs)} at {config.attn_layer_idxs}")
        print(f"  SSM layers: {len(config.ssm_layer_idxs)} at {config.ssm_layer_idxs}")
        print(f"  RoPE: theta={config.rope_theta}, max_pos={config.max_position_embeddings}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
