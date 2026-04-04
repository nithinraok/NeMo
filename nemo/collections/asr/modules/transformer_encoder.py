from torch import nn
import torch
from torch.nn import Conv1d
from dataclasses import dataclass
from torch.nn import GELU as TorchGELU
from torch.nn.functional import scaled_dot_product_attention
@dataclass
class GPTConfig():
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: int = 0.1
    qkv_bias: bool = False
    theta_base: int = 10_000

@dataclass
class TransformerEncoderConfig():
    n_mels: int = 80
    d_model: int = 512
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False
    causal_mask: bool = False
    theta_base: int = 10_000
    context_length: int = 4096
    qk_norm: bool = False

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim=dim
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4*dim),
            TorchGELU(),
            nn.Linear(4*dim, dim)
        )

    def forward(self, x):
        return self.ffn(x)

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased = False)
        with torch.autocast('cuda', dtype=torch.float32):
            norm = (x-mean)/ torch.sqrt(var + self.eps)
            output = self.scale * norm + self.shift

        return output

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0) # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def apply_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2:]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)


class MultiHeadAttentionWithFA(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.0, qkv_bias=False, context_length=1024, num_heads=8, causal_mask=False):
        super().__init__()
        self.d_out = dim_out
        self.w_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.dropout = dropout
        self.causal_mask = causal_mask
        self.out_proj = nn.Linear(self.d_out, self.d_out)


    def forward(self, x):
        
        B, num_tokens, d_in  = x.shape
        H = self.num_heads

        keys = self.w_key(x).view(B,num_tokens,H, self.head_dim) # Bxnum_tokens x Hx head_dim
        queries = self.w_query(x).view(B,num_tokens,H, self.head_dim)
        values = self.w_value(x).view(B,num_tokens,H, self.head_dim)
        
        dropout = 0 if self.training == False else self.dropout        
        output = flash_attn_func(queries,keys,values, dropout_p=dropout, causal=self.causal_mask)

        # Bxnum_tokens x Hx head_dim

        output = output.contiguous().view(B,num_tokens,self.d_out)

        output = self.out_proj(output)

        return output
class MultiHeadAttentionWithSDPA(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.0, qkv_bias=False, context_length=1024, num_heads=8, causal_mask=False, qk_norm=False):
        super().__init__()
        self.d_out = dim_out
        self.w_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.dropout = dropout
        self.causal_mask = causal_mask
        self.out_proj = nn.Linear(self.d_out, self.d_out)
        self.qk_norm = qk_norm
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x, attn_mask=None, use_cache=False):

        B, num_tokens, d_in  = x.shape
        H = self.num_heads

        keys = self.w_key(x).view(B,num_tokens,H, self.head_dim) # Bxnum_tokens x Hx head_dim
        queries = self.w_query(x).view(B,num_tokens,H, self.head_dim)
        values = self.w_value(x).view(B,num_tokens,H, self.head_dim)

        keys = keys.transpose(1,2) # BxHxnum_tokens,head_dim
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        if self.qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        dropout = 0 if self.training == False else self.dropout
        output = scaled_dot_product_attention(queries,keys,values, attn_mask=attn_mask, is_causal=self.causal_mask, dropout_p=dropout)

         # B xH x num_tokens x head_dim

        output = output.transpose(1,2) # Bxnum_tokens x Hx head_dim

        output = output.contiguous().view(B,num_tokens,self.d_out)

        output = self.out_proj(output)

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.0, qkv_bias=False, context_length=1024, num_heads=8, causal_mask=False):
        super().__init__()
        self.d_out = dim_out
        self.w_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.dropout = nn.Dropout(dropout)
        self.causal_mask = causal_mask
        self.out_proj = nn.Linear(self.d_out, self.d_out)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1) if self.causal_mask else torch.zeros(context_length, context_length)
        )
        
        self.register_buffer(
            "cache_k", None, persistent=False
        )
        self.register_buffer(
            "cache_v", None, persistent=False
        )


    def forward(self, x, use_cache=False):
        
        B, num_tokens, d_in  = x.shape
        H = self.num_heads

        keys = self.w_key(x).view(B,num_tokens,H, self.head_dim) # Bxnum_tokens x Hx head_dim
        queries = self.w_query(x).view(B,num_tokens,H, self.head_dim)
        values = self.w_value(x).view(B,num_tokens,H, self.head_dim)

        if use_cache:
            if self.cache_k is None:
                self.cache_k  = keys
                self.cache_v = values
            else:
                self.cache_k = torch.cat((self.cache_k, keys), dim=1)
                self.cache_v = torch.cat((self.cache_v, values), dim=1)
            keys, values = self.cache_k, self.cache_v


        keys = keys.transpose(1,2) # BxHxnum_tokens,head_dim
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        attn_scores = torch.matmul(queries, keys.transpose(-1,-2)) # We need to transpose head_dim and num_tokens for keys. alpha = BxNxTqxTk
        d_k = keys.shape[-1]

        # Masking 
        T = attn_scores.shape[-1]
        mask = self.mask[:num_tokens, :num_tokens]
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)

        attn_weights = torch.softmax(masked/d_k**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, values) # B xH x num_tokens x head_dim

        output = output.transpose(1,2) # Bxnum_tokens x Hx head_dim

        output = output.contiguous().view(B,num_tokens,self.d_out)

        output = self.out_proj(output)

        return output
    
    def reset_cache(self,):
        self.cache_k = None
        self.cache_v = None



class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.pre_norm = LayerNorm(self.cfg.d_model)
        self.mha = MultiHeadAttentionWithSDPA(
            dim_in=self.cfg.d_model,
            dim_out=self.cfg.d_model,
            dropout=self.cfg.drop_rate,
            qkv_bias=self.cfg.qkv_bias,
            num_heads=self.cfg.n_heads,
            causal_mask=self.cfg.causal_mask,
            qk_norm=self.cfg.qk_norm
            )
        self.dropout = nn.Dropout(self.cfg.drop_rate)
        self.post_norm = LayerNorm(self.cfg.d_model)
        self.ffn = FeedForward(self.cfg.d_model)

    def forward(self, x, attn_mask=None, use_cache=False):
        pre_norm = self.pre_norm(x)

        attn_output = self.mha(pre_norm, attn_mask=attn_mask, use_cache=use_cache)
        attn_output = x + self.dropout(attn_output)

        post_norm = self.post_norm(attn_output)
        ffn = self.ffn(post_norm)
        output = attn_output + self.dropout(ffn)
        return output

class ConvSubsampling(nn.Module):
    def __init__(self, n_mels: int = 80, d_model: int = 512):
        super().__init__()
        self.conv1 = Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1) # Decreases the temporal dimension by 2
        self.conv3 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1) # Decreases the temporal dimension by 2
        self.gelu = TorchGELU()

    def forward(self, x, length):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)
        length = length // 2
        x = self.conv3(x)
        x = self.gelu(x)
        length = length // 2
        x = x.transpose(1, 2)  # (B, d_model, T) -> (B, T, d_model)
        return x, length

class DepthwiseConvSubsampling(nn.Module):
    """Depthwise separable conv subsampling: reduces params by replacing standard Conv1d
    with depthwise (groups=channels) + pointwise (1x1) convolutions for the strided layers.
    """
    def __init__(self, n_mels: int = 80, d_model: int = 512):
        super().__init__()
        # Standard conv to project from n_mels to d_model
        self.conv1 = Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        # Depthwise separable conv block 1 (stride=2)
        self.dw_conv2 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, groups=d_model)
        self.pw_conv2 = Conv1d(d_model, d_model, kernel_size=1)
        # Depthwise separable conv block 2 (stride=2)
        self.dw_conv3 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, groups=d_model)
        self.pw_conv3 = Conv1d(d_model, d_model, kernel_size=1)
        self.gelu = TorchGELU()

    def forward(self, x, length):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.dw_conv2(x)
        x = self.pw_conv2(x)
        x = self.gelu(x)
        length = length // 2
        x = self.dw_conv3(x)
        x = self.pw_conv3(x)
        x = self.gelu(x)
        length = length // 2
        x = x.transpose(1, 2)  # (B, d_model, T) -> (B, T, d_model)
        return x, length

class NGPTStackingSubsampling(torch.nn.Module):
    """Stacking subsampling which simply stacks consecutive frames to reduce the sampling rate
    Args:
        subsampling_factor (int): The subsampling factor
        feat_in (int): size of the input features
        feat_out (int): size of the output features
    """

    def __init__(
        self,
        subsampling_factor: int,
        feat_in: int,
        feat_out: int,
        use_bias: bool = False
    ):
        super().__init__()
        self.subsampling_factor = subsampling_factor
        self.proj_out = torch.nn.Linear(subsampling_factor * feat_in, feat_out, bias=use_bias)
        self.pad_frame = nn.Parameter(torch.ones(feat_in, dtype=torch.float32))

    def forward(self, x, length):
        """
        Args:
            x (torch.Tensor): (B, C, T)
            length (torch.Tensor): (B,)
        Returns:
            x (torch.Tensor): (B, T', D_model)
            length (torch.Tensor): (B,)
        """
        x = x.transpose(1, 2) # BxCxT -> BxTxC
        b, t, h = x.size()
        pad_size = (self.subsampling_factor - (t % self.subsampling_factor)) % self.subsampling_factor
        length = torch.div(length + pad_size, self.subsampling_factor, rounding_mode='floor')

        # Pad and fill padding frames (all-zero) with a learnable padding 'embedding'
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        x[(x == 0).all(dim=-1)] = self.pad_frame

        _, t, _ = x.size()
        x = torch.reshape(x, (b, t // self.subsampling_factor, h * self.subsampling_factor))
        x = self.proj_out(x)

        return x, length
class TransformerEncoder(nn.Module):
    def __init__(self,
                n_mels: int = 80,
                d_model: int = 512,
                n_heads: int = 8,
                n_layers: int = 17,
                drop_rate: float = 0.1,
                qkv_bias: bool = False,
                causal_mask: bool = False,
                pre_encode: str = "conv", # "conv" or "stacking"
                nan_debug: bool = True,
                qk_norm: bool = False,
                subsampling_factor: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.nan_debug = nan_debug
        if pre_encode == "conv":
            self.pre_encode = ConvSubsampling(n_mels, d_model)
        elif pre_encode == "depth_conv":
            self.pre_encode = DepthwiseConvSubsampling(n_mels, d_model)
        elif pre_encode == "stacking":
            self.pre_encode = NGPTStackingSubsampling(subsampling_factor=subsampling_factor, feat_in=n_mels, feat_out=d_model)
        else:
            raise ValueError(f"Invalid pre_encode: {pre_encode}. Choose from: conv, depth_conv, stacking")

        cfg = TransformerEncoderConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, drop_rate=drop_rate, qkv_bias=qkv_bias, causal_mask=causal_mask, qk_norm=qk_norm)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, audio_signal, length):
        """
        Args:
            audio_signal (torch.Tensor): (B, C, T)
            length (torch.Tensor): (B,)
        Returns:
            x (torch.Tensor): (B, T', D_model)
            length (torch.Tensor): (B,)
        """
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

        # Create padding mask: True for valid positions, False for padding
        max_len = x.shape[1]
        pad_mask = torch.arange(max_len, device=x.device).unsqueeze(0) < length.unsqueeze(1)
        # Convert to attention mask for SDPA: (B, 1, 1, T) broadcastable to (B, H, T, T)
        # SDPA treats True as "attend" for boolean masks
        attn_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T) - masks key positions

        for idx, layer in enumerate(self.layers):
            x = layer(x, attn_mask=attn_mask)
            if self.nan_debug:
                self._check_nan(x, f"layer_{idx}")
        x = self.final_norm(x)
        if self.nan_debug:
            self._check_nan(x, "final_norm")
        x = x.transpose(1, 2) # BxT'xD_model -> BxD_modelxT'
        return x, length

    def _check_nan(self, x, name):
        has_nan = torch.isnan(x).any().item()
        has_inf = torch.isinf(x).any().item()
        if has_nan or has_inf:
            nan_count = torch.isnan(x).sum().item()
            inf_count = torch.isinf(x).sum().item()
            valid = x[~(torch.isnan(x) | torch.isinf(x))]
            abs_max = valid.abs().max().item() if valid.numel() > 0 else float('nan')
            print(f"[NaN DEBUG] {name}: NaN={nan_count}, Inf={inf_count}, abs_max={abs_max:.6f}, shape={list(x.shape)}", flush=True)
            raise RuntimeError(f"[NaN DEBUG] NaN/Inf detected at '{name}'. Stopping training.")
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self, partial=False):
        for param in self.parameters():
            param.requires_grad = True


