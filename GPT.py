import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from torch import Tensor
from typing import Callable

torch.manual_seed(69420)


@dataclass
class GPTConfig:
    # Learning params
    batch_size: int         = 64
    max_iters: int          = 5000
    eval_interval: int      = 500
    device: str             = 'cuda'
    eval_iters: int         = 200
    # Optimizer params
    learning_rate: float    = 3e-4
    # Transformer params
    block_size: int         = 256   # sequence length
    n_embed: int            = 64
    n_head: int             = 8
    n_layer: int            = 6
    # Regularizer params
    dropout: float          = 0.2
    elem_aff: bool          = False
    ln_eps: float           = 1e-6
    ln_bias: bool           = True
    # Feedforward params
    ff_dim: int             = 4 * n_embed
    kqv_mlp: bool           = False
    ff_bias: bool           = True
    slow_attention: bool    = False
    masked: bool            = True
    

class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig, activation='ReLU'):
        super().__init__()
        self.activ_fn   = getattr(nn,activation)
        self.lin_1      = nn.Linear(cfg.n_embed, cfg.ff_dim)
        self.lin_2      = nn.Linear(cfg.ff_dim, cfg.n_embed)
        self.dropout    = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin_1(x)
        x = self.activ_fn(x)
        x = self.lin_2(x)
        x = self.dropout(x)
        return x
    

class CausalMHA(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()  
        assert cfg.n_embed % cfg.n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.cfg = cfg
        self.n_embed    = cfg.n_embed
        self.n_head     = cfg.n_head
        self.dropout    = cfg.dropout
        self.masked     = cfg.masked

        # Initialize the linear layers for query, key, and value
        if cfg.kqv_mlp:
            self.w_q = nn.Linear(cfg.n_embed, cfg.n_embed, bias=cfg.ff_bias)
            self.w_k = nn.Linear(cfg.n_embed, cfg.n_embed, bias=cfg.ff_bias)
            self.w_v = nn.Linear(cfg.n_embed, cfg.n_embed, bias=cfg.ff_bias)

        # Output projection layer
        self.out_proj       = nn.Linear(cfg.n_embed, cfg.n_embed, bias=cfg.ff_bias)
        self.attn_dropout   = nn.Dropout(cfg.dropout)
        self.resid_dropout  = nn.Dropout(cfg.dropout)
        
        self.flash = hasattr(torch.functional, 'scaled_dot_product_attention')
        if not self.flash or cfg.slow_attention: 
            print("Get PyTorch >= 2.0 for Flash Attention")
            self.register_buffer("bias", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                                 .view(1,1,cfg.block_size,cfg.block_size))

    def scaled_dot_product_attn(self, q:Tensor, k:Tensor, v:Tensor) -> Tensor:
        assert q.size() == k.size() == v.size(), "Query, Key, and Value must have the same shape"
        B, T, C = q.size()

        if self.flash:
            # efficient attnetino using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(q, k, v,
                                               attn_mask=None,
                                               dropout_p=self.dropout if self.training else 0, 
                                               is_causal=self.masked,)
        else:
            # manula implementation of attention
            att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.masked: att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (b, nh, T, hs)
        return y

    def split_heads(self, x:Tensor) -> Tensor:
        B, T, C = x.size()
        return x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    
    def combine_heads(self, x: Tensor) -> Tensor:
        B, nh, T, hs = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, nh * hs)

    def forward(self, q:Tensor, k:Tensor, v:Tensor) -> Tensor:
        assert q.size() == k.size() == v.size(), "Query, Key, and Value must have the same shape"
        B, T, C = q.size()

        if self.cfg.kqv_mlp: 
            q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)

        attn = self.scaled_dot_product_attn(q, k, v)
        y = self.out_proj(self.combine_heads(attn))
        y = self.resid_dropout(y)
        return y
    

class AdaLNBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn = CausalMHA(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln_1 = nn.LayerNorm(cfg.n_embed, bias=cfg.ln_bias, 
                                 elementwise_affine=cfg.elem_aff, eps=cfg.ln_eps)
        self.ln_2 = nn.LayerNorm(cfg.n_embed, bias=cfg.ln_bias, 
                                 elementwise_affine=cfg.elem_aff, eps=cfg.ln_eps)
        self.adaLN_modulation: Callable[[Tensor], Tensor] = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(cfg.n_embed, 6 * cfg.n_embed), bias=cfg.ff_bias
            )
    
    @staticmethod
    def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale) + shift

    def forward(self, x: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_ffwd, scale_ffwd, gate_ffwd = \
            self.adaLN_modulation(x).chunk(6, dim=-1)
        moduln = self.modulate(self.ln_1(x), shift_msa, scale_msa)
        x = x + self.attn(moduln, moduln, moduln) * gate_msa.unsqueeze(-1)
        x = x + self.ffwd(self.modulate(self.ln_2(x), shift_ffwd, scale_ffwd)) * gate_ffwd.unsqueeze(-1)
        return x
    

class GPTBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.self_attn = CausalMHA(cfg)
        self.cross_attn = CausalMHA(cfg)
        self.ffwd = FeedForward(cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        self.ln_1 = nn.LayerNorm(cfg.n_embed, bias=cfg.ln_bias, 
                                 elementwise_affine=cfg.elem_aff, eps=cfg.ln_eps)
        self.ln_2 = nn.LayerNorm(cfg.n_embed, bias=cfg.ln_bias, 
                                 elementwise_affine=cfg.elem_aff, eps=cfg.ln_eps)
        self.ln_3 = nn.LayerNorm(cfg.n_embed, bias=cfg.ln_bias, 
                                 elementwise_affine=cfg.elem_aff, eps=cfg.ln_eps)
        
    def forward(self, x: Tensor, context: Tensor = None) -> Tensor:
        x = self.ln_1(x)
        attn = self.self_attn(x, x, x)
        x = self.ln_2(x + self.dropout(attn))
        attn = self.cross_attn(x, context, context)
        x = self.ln_3(x + self.dropout(attn))
        x = x + self.dropout(self.ffwd(x))
        return x