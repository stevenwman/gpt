import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from torch import Tensor
from typing import Callable

torch.manual_seed(69420)


@dataclass
class Config:
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
    n_heads: int             = 8
    n_layers: int            = 6
    state_dim: int          = 7
    action_dim: int         = 6
    # Regularizer params
    dropout: float          = 0.2
    elem_aff: bool          = False # only for AdaLN
    ln_eps: float           = 1e-6  # only for AadLN
    ln_bias: bool           = True  # only for AdaLN
    # Feedforward params
    ff_dim: int             = 4 * n_embed
    kqv_mlp: bool           = False
    ff_bias: bool           = True
    slow_attention: bool    = False
    masked: bool            = True


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """
    Shift scale operations for AdaLN
    """
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, cfg: Config, activation='ReLU'):
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
    def __init__(self, cfg: Config):
        super().__init__()  
        assert cfg.n_embed % cfg.n_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.cfg = cfg
        self.n_embed    = cfg.n_embed
        self.n_heads     = cfg.n_heads
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
        return x.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
    
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
    def __init__(self, cfg: Config):
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

    def forward(self, x: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_ffwd, scale_ffwd, gate_ffwd = \
            self.adaLN_modulation(x).chunk(6, dim=-1)
        moduln = modulate(self.ln_1(x), shift_msa, scale_msa)
        x = x + self.attn(moduln, moduln, moduln) * gate_msa.unsqueeze(-1)
        x = x + self.ffwd(modulate(self.ln_2(x), shift_ffwd, scale_ffwd)) * gate_ffwd.unsqueeze(-1)
        return x
    

class GPTBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.self_attn = CausalMHA(cfg)
        self.cross_attn = CausalMHA(cfg)
        self.ffwd = FeedForward(cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        self.ln_1 = nn.LayerNorm(cfg.n_embed)
        self.ln_2 = nn.LayerNorm(cfg.n_embed)
        self.ln_3 = nn.LayerNorm(cfg.n_embed)
        
    def forward(self, x: Tensor, context: Tensor = None) -> Tensor:
        x = self.ln_1(x)
        attn = self.self_attn(x, x, x)
        x = self.ln_2(x + self.dropout(attn))
        attn = self.cross_attn(x, context, context)
        x = self.ln_3(x + self.dropout(attn))
        x = x + self.dropout(self.ffwd(x))
        return x
    

class AdaLNFinalLayer(nn.Module):
    def __init__(self, cfg: Config, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(cfg.n_embed, elementwise_affine=cfg.elem_aff, eps=cfg.ln_eps)
        self.linear = nn.Linear(cfg.n_embed, out_dim, bias=cfg.ff_bias)
        self.adaLN_modulation: Callable[[Tensor], Tensor] = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.n_embed, 2 * cfg.n_embed, bias=cfg.ff_bias)
        )

    def forward(self, inp, cond):
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=2)
        inp = modulate(self.norm_final(inp), shift, scale)
        out = self.linear(inp)
        return out
    

class GPT_XAttn(nn.Module):
    def __init__(self, cfg: Config, pos_embedding: Callable[[Tensor],Tensor], is_critic: bool):
        self.encode_state = nn.Linear(cfg.state_dim, cfg.n_embed)
        self.embed_action = nn.Linear(cfg.action_dim, cfg.n_embed)
        self.pos_embedding = pos_embedding
        self.dropout = nn.Dropout(cfg.dropout)
        self.decoder_layers = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.n_layers)])

        out_dim = 1 if is_critic else cfg.action_dim
        self.actor_mu_layer = nn.Linear(cfg.n_embed, out_dim)
        self.activation = nn.GELU()

    def forward(self, state, actions, pos):
        state_enc = self.encode_state(state)
        pos_embed = self.pos_embedding(pos)
        cond_enc = pos_embed.squeeze(2) + state_enc
        act_enc = self.activation(self.embed_action(actions))
        for layer in self.decoder_layers:
            act_enc = layer(act_enc, cond_enc)
        act_mean = self.actor_mu_layer(act_enc)
        return act_mean
    

class GPT_AdaLN(nn.Module):
    def __init__(self, cfg: Config, pos_embedding: Callable[[Tensor],Tensor], is_critic: bool):
        super().__init__()
        self.encode_state = nn.Linear(cfg.state_dim, cfg.n_embed)
        self.embed_action = nn.Linear(cfg.action_dim, cfg.n_embed)
        self.pos_embedding = pos_embedding
        self.dropout = nn.Dropout(cfg.dropout)

        self.decoder_layers = nn.ModuleList([AdaLNBlock(cfg) for _ in range(cfg.n_layers)])
        out_dim = 1 if is_critic else cfg.action_dim
        self.final_layer = AdaLNFinalLayer(cfg, out_dim)
        self.activation = nn.GELU

    def forward(self, state, actions, pos):
        state_enc = self.encode_state(state)
        pos_embed = self.pos_embedding(pos)
        cond_enc = pos_embed.squeeze(2) + state_enc
        act_enc = self.activation(self.embed_action(actions))
        for layer in self.decoder_layers:
            act_enc = layer(act_enc, cond_enc)
        act_mean = self.final_layer(act_enc, cond_enc)
        return act_mean
    

class IntegerEmbeddingModel(nn.Module):
    def __init__(self, num_embeddings, n_embed):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, n_embed)
        self.linear1 = nn.Linear(n_embed, n_embed)
        self.linear2 = nn.Linear(n_embed, n_embed)
    
    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: Config, act_limit, use_adaln: bool):
        self.cfg = cfg
        self.device = cfg.device
        self.act_limit = act_limit
        self.act_dim = cfg.action_dim
        self.pos_embedding = IntegerEmbeddingModel(self.act_dim, cfg.n_embed)
        log_std = -0.5 * torch.ones(self.act_dim)
        self.log_std = nn.Parameter(log_std)

        GPTArch = GPT_AdaLN if use_adaln else GPT_XAttn
        self.decoder_actor = GPTArch(cfg, self.pos_embedding, is_critic=False)
        self.decoder_critic1 = GPTArch(cfg, self.pos_embedding, is_critic=True)
        self.decoder_critic2 = GPTArch(cfg, self.pos_embedding, is_critic=True)

    # TODO: figure out how you want to init weights
    # might just go with outside function
    def _init_weights(self):
        def _basic_init(module, activation="relu"):
            nn.init.orthogonal_(module, gain=nn.init.calculate_gain(activation))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

