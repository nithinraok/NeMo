"""
TransformerEncoder with FlexAttention support.

Supports multiple attention modes via a single `attn_mode` config parameter:
  - "full"           : bidirectional (non-causal), equivalent to current SDPA encoder
  - "causal"         : strictly causal, attend only to past + current
  - "lookahead"      : causal + N future frames (set `lookahead`)
  - "local"          : bidirectional window [left_ctx, right_ctx] (set `left_ctx`, `right_ctx`)
  - "sliding_window" : causal + limited history (set `window_size`)
  - "sliding window + global" -> local for some layers and global for some layers 4:1 ratio (gemma 4 implementation)

FlexAttention compiles mask_mod into block-sparse Triton kernels, skipping
masked tiles entirely (free compute savings for sparse patterns).

Note: flex_attention requires bf16/fp16 on Ampere (head_dim >= 80 exceeds
shared memory in fp32). On Hopper (H100), fp32 also works.
"""
from torch import nn
import torch
from torch.nn import Conv1d
from dataclasses import dataclass
from torch.nn import GELU as TorchGELU
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    and_masks,
)

# Compile flex_attention once at module level for fused Triton kernels.
# dynamic=True: generates shape-generic Triton kernels so variable (B, T) from
# Lhotse bucketed batching does not trigger repeated recompilation.
flex_attention_compiled = torch.compile(flex_attention, dynamic=True)


# ── Mask modifier builders ──────────────────────────────────────────────
# Each returns a mask_mod(b, h, q_idx, kv_idx) -> bool scalar.
# These are traced by torch.compile — keep them simple (no Python control flow).

def _make_full_mod():
    """Full bidirectional — no position-based masking."""
    def full_mask(b, h, q_idx, kv_idx):
        return q_idx >= 0  # always True
    return full_mask

def _make_causal_mod():
    """Strictly causal — attend only to past and current position."""
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    return causal

def _make_lookahead_mod(lookahead: int):
    """Causal with limited future lookahead of `lookahead` frames."""
    def lookahead_mask(b, h, q_idx, kv_idx):
        return kv_idx <= q_idx + lookahead
    return lookahead_mask

def _make_local_window_mod(left_ctx: int, right_ctx: int):
    """Local bidirectional window — attend to [q - left_ctx, q + right_ctx]."""
    def local_window(b, h, q_idx, kv_idx):
        return (kv_idx >= q_idx - left_ctx) & (kv_idx <= q_idx + right_ctx)
    return local_window

def _make_sliding_window_causal_mod(window: int):
    """Causal + sliding window — attend to past `window` positions only."""
    def sliding(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx <= window)
    return sliding

def _make_padding_mod(lengths):
    """Mask out padding positions based on per-sample lengths."""
    def pad_mask(b, h, q_idx, kv_idx):
        return kv_idx < lengths[b]
    return pad_mask

def build_attn_mask_mod(attn_mode, lookahead=0, left_ctx=0, right_ctx=0, window_size=0):
    """Build attention pattern mask_mod from config. Called once at __init__."""
    if attn_mode == "full":
        return _make_full_mod()
    elif attn_mode == "causal":
        return _make_causal_mod()
    elif attn_mode == "lookahead":
        assert lookahead > 0, f"lookahead must be > 0 for attn_mode='lookahead', got {lookahead}"
        return _make_lookahead_mod(lookahead)
    elif attn_mode == "local":
        assert left_ctx > 0 or right_ctx > 0, "left_ctx or right_ctx must be > 0 for attn_mode='local'"
        return _make_local_window_mod(left_ctx, right_ctx)
    elif attn_mode == "sliding_window":
        assert window_size > 0, f"window_size must be > 0 for attn_mode='sliding_window', got {window_size}"
        return _make_sliding_window_causal_mod(window_size)
    else:
        raise ValueError(
            f"Unknown attn_mode: '{attn_mode}'. "
            f"Choose from: full, causal, lookahead, local, sliding_window"
        )


# ── Config ──────────────────────────────────────────────────────────────

@dataclass
class TransformerEncoderConfig():
    n_mels: int = 80
    d_model: int = 512
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False
    qk_norm: bool = False
    # Attention mode
    attn_mode: str = "full"
    lookahead: int = 0
    left_ctx: int = 0
    right_ctx: int = 0
    window_size: int = 0


# ── Layers ──────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            TorchGELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        return self.ffn(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        with torch.autocast('cuda', dtype=torch.float32):
            norm = (x - mean) / torch.sqrt(var + self.eps)
            return self.scale * norm + self.shift


class MultiHeadAttentionWithFlex(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.0, qkv_bias=False, num_heads=8, qk_norm=False):
        super().__init__()
        self.d_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.dropout = dropout

        self.w_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.out_proj = nn.Linear(dim_out, dim_out)

        self.qk_norm = qk_norm
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x, block_mask=None):
        B, T, _ = x.shape
        H = self.num_heads

        queries = self.w_query(x).view(B, T, H, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        keys = self.w_key(x).view(B, T, H, self.head_dim).transpose(1, 2)
        values = self.w_value(x).view(B, T, H, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            queries = self.q_norm(queries).to(values.dtype)
            keys = self.k_norm(keys).to(values.dtype)

        output = flex_attention_compiled(queries, keys, values, block_mask=block_mask)

        output = output.transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.out_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerEncoderConfig):
        super().__init__()
        self.pre_norm = LayerNorm(cfg.d_model)
        self.mha = MultiHeadAttentionWithFlex(
            dim_in=cfg.d_model,
            dim_out=cfg.d_model,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias,
            num_heads=cfg.n_heads,
            qk_norm=cfg.qk_norm,
        )
        self.dropout = nn.Dropout(cfg.drop_rate)
        self.post_norm = LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model)

    def forward(self, x, block_mask=None):
        # Pre-norm MHA + residual
        h = self.mha(self.pre_norm(x), block_mask=block_mask)
        x = x + self.dropout(h)
        # Pre-norm FFN + residual
        h = self.ffn(self.post_norm(x))
        x = x + self.dropout(h)
        return x


# ── Pre-encoder modules ────────────────────────────────────────────────

class ConvSubsampling(nn.Module):
    def __init__(self, n_mels: int = 80, d_model: int = 512):
        super().__init__()
        self.conv1 = Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.gelu = TorchGELU()

    def forward(self, x, length):
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        length = length // 2
        x = self.gelu(self.conv3(x))
        length = length // 2
        x = x.transpose(1, 2)  # (B, d_model, T) -> (B, T, d_model)
        return x, length


class DepthwiseConvSubsampling(nn.Module):
    def __init__(self, n_mels: int = 80, d_model: int = 512):
        super().__init__()
        self.conv1 = Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.dw_conv2 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, groups=d_model)
        self.pw_conv2 = Conv1d(d_model, d_model, kernel_size=1)
        self.dw_conv3 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, groups=d_model)
        self.pw_conv3 = Conv1d(d_model, d_model, kernel_size=1)
        self.gelu = TorchGELU()

    def forward(self, x, length):
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.pw_conv2(self.dw_conv2(x)))
        length = length // 2
        x = self.gelu(self.pw_conv3(self.dw_conv3(x)))
        length = length // 2
        x = x.transpose(1, 2)
        return x, length


class NGPTStackingSubsampling(nn.Module):
    def __init__(self, subsampling_factor: int, feat_in: int, feat_out: int, use_bias: bool = False):
        super().__init__()
        self.subsampling_factor = subsampling_factor
        self.proj_out = nn.Linear(subsampling_factor * feat_in, feat_out, bias=use_bias)
        self.pad_frame = nn.Parameter(torch.ones(feat_in, dtype=torch.float32))

    def forward(self, x, length):
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        b, t, h = x.size()
        pad_size = (self.subsampling_factor - (t % self.subsampling_factor)) % self.subsampling_factor
        length = torch.div(length + pad_size, self.subsampling_factor, rounding_mode='floor')

        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        x[(x == 0).all(dim=-1)] = self.pad_frame

        _, t, _ = x.size()
        x = torch.reshape(x, (b, t // self.subsampling_factor, h * self.subsampling_factor))
        x = self.proj_out(x)
        return x, length


# ── Encoder ─────────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 17,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
        pre_encode: str = "conv",
        nan_debug: bool = True,
        qk_norm: bool = False,
        subsampling_factor: int = 4,
        # Flex attention config
        attn_mode: str = "full",
        lookahead: int = 0,
        left_ctx: int = 0,
        right_ctx: int = 0,
        window_size: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.nan_debug = nan_debug
        self.attn_mode = attn_mode

        # Pre-encoder
        if pre_encode == "conv":
            self.pre_encode = ConvSubsampling(n_mels, d_model)
        elif pre_encode == "depth_conv":
            self.pre_encode = DepthwiseConvSubsampling(n_mels, d_model)
        elif pre_encode == "stacking":
            self.pre_encode = NGPTStackingSubsampling(
                subsampling_factor=subsampling_factor, feat_in=n_mels, feat_out=d_model,
            )
        else:
            raise ValueError(f"Invalid pre_encode: {pre_encode}. Choose from: conv, depth_conv, stacking")

        # Attention pattern mask (fixed at init, combined with padding at forward time)
        self.attn_mask_mod = build_attn_mask_mod(
            attn_mode=attn_mode,
            lookahead=lookahead,
            left_ctx=left_ctx,
            right_ctx=right_ctx,
            window_size=window_size,
        )

        # Transformer layers
        cfg = TransformerEncoderConfig(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            drop_rate=drop_rate, qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_mode=attn_mode, lookahead=lookahead,
            left_ctx=left_ctx, right_ctx=right_ctx, window_size=window_size,
        )
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, audio_signal, length):
        x = audio_signal
        x, length = self.pre_encode(x, length)
        if self.nan_debug:
            self._check_nan(x, "pre_encode")

        x = x * (self.d_model ** 0.5)
        if self.nan_debug:
            self._check_nan(x, "embedding_scale")

        x = self.layer_norm(x)
        if self.nan_debug:
            self._check_nan(x, "layer_norm")

        # Build block mask: attention pattern AND padding, shared across all layers
        B, T, _ = x.shape
        padding_mod = _make_padding_mod(length)
        combined_mod = and_masks(self.attn_mask_mod, padding_mod)
        block_mask = create_block_mask(combined_mod, B=B, H=1, Q_LEN=T, KV_LEN=T, device=x.device)

        for idx, layer in enumerate(self.layers):
            x = layer(x, block_mask=block_mask)
            if self.nan_debug:
                self._check_nan(x, f"layer_{idx}")

        x = self.final_norm(x)
        if self.nan_debug:
            self._check_nan(x, "final_norm")

        x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T)
        return x, length

    def _check_nan(self, x, name):
        has_nan = torch.isnan(x).any().item()
        has_inf = torch.isinf(x).any().item()
        if has_nan or has_inf:
            nan_count = torch.isnan(x).sum().item()
            inf_count = torch.isinf(x).sum().item()
            valid = x[~(torch.isnan(x) | torch.isinf(x))]
            abs_max = valid.abs().max().item() if valid.numel() > 0 else float('nan')
            print(
                f"[NaN DEBUG] {name}: NaN={nan_count}, Inf={inf_count}, "
                f"abs_max={abs_max:.6f}, shape={list(x.shape)}",
                flush=True,
            )
            raise RuntimeError(f"[NaN DEBUG] NaN/Inf detected at '{name}'. Stopping training.")

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self, partial=False):
        for param in self.parameters():
            param.requires_grad = True
