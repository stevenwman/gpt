import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from torch import Tensor
from typing import Callable, cast

torch.manual_seed(69420)


@dataclass
class Config:
    # Learning params
    batch_size: int         = 6
    max_iters: int          = 5000
    eval_interval: int      = 500
    device: str             = 'cuda'
    eval_iters: int         = 200
    seed: int               = 69420
    # Optimizer params
    learning_rate: float    = 3e-4
    # Transformer params
    n_embed: int            = 10
    n_heads: int            = 5
    n_layers: int           = 4
    state_dim: int          = 3     # conditioning variables sequence length
    action_dim: int         = 2     # input sequence length
    pos_embed: bool         = True  # use positional embedding
    # Regularizer params
    dropout: float          = 0.2
    elem_aff: bool          = False # only for AdaLN
    ln_eps: float           = 1e-6  # only for AadLN
    ln_bias: bool           = True  # only for AdaLN
    # Feedforward params
    ff_dim: int             = 4 * n_embed
    kqv_mlp: bool           = True
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
        self.activ_fn   = getattr(nn,activation)()
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
        
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash or cfg.slow_attention: 
            print("Get PyTorch >= 2.0 for Flash Attention")
            self.register_buffer("bias", torch.tril(torch.ones(cfg.action_dim, cfg.action_dim))
                                 .view(1,1,cfg.action_dim,cfg.action_dim))

    def scaled_dot_product_attn(self, q:Tensor, k:Tensor, v:Tensor) -> Tensor:
        assert q.size() == k.size() == v.size(), "Query, Key, and Value must have the same shape"
        B, T, C  = q.size()
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)

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

        y = self.combine_heads(y)  # (B, T, nh * hs)
        return y

    def split_heads(self, x:Tensor) -> Tensor:
        B, T, C = x.size()
        return x.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
    
    def combine_heads(self, x: Tensor) -> Tensor:
        B, nh, T, hs = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, nh * hs)

    def forward(self, q:Tensor, k:Tensor, v:Tensor) -> Tensor:
        assert q.size() == k.size() == v.size(), "Query, Key, and Value must have the same shape"

        if self.cfg.kqv_mlp: 
            q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        attn = self.scaled_dot_product_attn(q, k, v)
        y = self.out_proj(attn)
        y = self.resid_dropout(y)
        return y
    

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
        
    def forward(self, x: Tensor, cond: Tensor = None) -> Tensor:
        x = self.ln_1(x)
        attn = self.self_attn(x, x, x)
        x = self.ln_2(x + self.dropout(attn))
        attn = self.cross_attn(x, cond, cond)
        x = self.ln_3(x + self.dropout(attn))
        x = x + self.dropout(self.ffwd(x))
        return x


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
            nn.Linear(cfg.n_embed, 6 * cfg.n_embed, bias=cfg.ff_bias)
            )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_ffwd, scale_ffwd, gate_ffwd = \
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        moduln = modulate(self.ln_1(x), shift_msa, scale_msa)
        x = x + self.attn(moduln, moduln, moduln) * gate_msa
        x = x + self.ffwd(modulate(self.ln_2(x), shift_ffwd, scale_ffwd)) * gate_ffwd
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

    def forward(self, inp: Tensor, cond: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=2)
        inp = modulate(self.norm_final(inp), shift, scale)
        out = self.linear(inp)
        return out
    

class GPT_XAttn(nn.Module):
    def __init__(self, cfg: Config, is_critic: bool):
        super().__init__()
        self.encode_state = nn.Linear(cfg.state_dim, cfg.n_embed)
        self.embed_action = nn.Linear(cfg.action_dim, cfg.n_embed)
        self.cfg = cfg

        if cfg.pos_embed:
            self.act_pos_embedding = IntegerEmbeddingModel(cfg.action_dim, cfg.n_embed)
            self.cond_pos_embedding = IntegerEmbeddingModel(cfg.state_dim, cfg.n_embed)
        self.dropout = nn.Dropout(cfg.dropout)
        self.decoder_layers = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.n_layers)])

        out_dim = 1 if is_critic else cfg.action_dim
        self.actor_mu_layer = nn.Linear(cfg.n_embed, out_dim)
        self.activation = nn.GELU()

    def forward(self, state, actions):
        state_enc = self.encode_state(state)
        act_enc = self.embed_action(actions)

        if self.cfg.pos_embed:
            act_pos_embed = self.act_pos_embedding(actions)
            state_pos_embed = self.cond_pos_embedding(state)
            cond_enc = state_pos_embed + state_enc
            act_enc = act_pos_embed + self.activation(act_enc)

        for layer in self.decoder_layers:
            act_enc = layer(act_enc, cond_enc)
        act_mean = self.actor_mu_layer(act_enc)
        return act_mean
    

class GPT_AdaLN(nn.Module):
    def __init__(self, cfg: Config, is_critic: bool):
        super().__init__()
        self.cfg = cfg

        self.encode_state: Callable[[Tensor], Tensor] = nn.Linear(cfg.state_dim, cfg.n_embed)
        self.embed_action: Callable[[Tensor], Tensor] = nn.Linear(cfg.action_dim, cfg.n_embed)
        # fullshit moment
        self.state_compressor: Callable[[Tensor], Tensor] = nn.Linear(cfg.state_dim, cfg.action_dim)

        if cfg.pos_embed:
            self.act_pos_embedding = IntegerEmbeddingModel(cfg.action_dim, cfg.n_embed)
            self.cond_pos_embedding = IntegerEmbeddingModel(cfg.state_dim, cfg.n_embed)

        self.dropout = nn.Dropout(cfg.dropout)
        self.decoder_layers = nn.ModuleList([AdaLNBlock(cfg) for _ in range(cfg.n_layers)])

        out_dim = 1 if is_critic else cfg.action_dim
        self.final_layer = AdaLNFinalLayer(cfg, out_dim)
        self.activation = nn.GELU()

    def forward(self, state: Tensor, actions: Tensor) -> Tensor:
        state_enc = self.encode_state(state.transpose(-1, -2)).squeeze(-1)  # (B, n_embed)
        act_enc = self.embed_action(actions.transpose(-1, -2)).squeeze(-1)  # (B, n_embed)

        if self.cfg.pos_embed:
            act_pos_embed = self.act_pos_embedding(actions)
            state_pos_embed = self.cond_pos_embedding(state)
            cond_enc: Tensor = state_pos_embed + state_enc
            cond_enc = self.state_compressor(cond_enc.transpose(-1,-2)).transpose(-1, -2)  # (B, n_embed, 1)
            act_enc = act_pos_embed + self.activation(act_enc)

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
    
    def forward(self, x: Tensor) -> Tensor:
        idx = torch.arange(x.size(1), device=x.device) # create 0-dim index tensor
        idx = self.embedding(idx)
        idx = F.relu(self.linear1(idx))
        idx = F.relu(self.linear2(idx))
        return idx


class Transformer(nn.Module):
    def __init__(self, cfg: Config, act_limit: Tensor, use_adaln: bool):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.act_limit = act_limit
        self.act_dim = cfg.action_dim
        log_std = -0.5 * torch.ones(self.act_dim)
        self.log_std = nn.Parameter(log_std)

        GPTArch = GPT_AdaLN if use_adaln else GPT_XAttn
        self.actor = GPTArch(cfg, is_critic=False)
        self.critic1 = GPTArch(cfg, is_critic=True)
        self.critic2 = GPTArch(cfg, is_critic=True)

        self.to(self.device)
        self._init_weights()

    def _init_weights(self):
        # Initialize transformer layers
        def _basic_init(module, activation="relu"):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain(activation))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def _qkv_mlp_init(module):
            if self.cfg.kqv_mlp and isinstance(module, CausalMHA):
                # Initialize query, key, value linear layers
                nn.init.orthogonal_(module.w_q.weight, gain=nn.init.calculate_gain('linear'))
                nn.init.orthogonal_(module.w_k.weight, gain=nn.init.calculate_gain('linear'))
                nn.init.orthogonal_(module.w_v.weight, gain=nn.init.calculate_gain('linear'))
                if module.w_q.bias is not None:
                    nn.init.constant_(module.w_q.bias, 0)
                    nn.init.constant_(module.w_k.bias, 0)
                    nn.init.constant_(module.w_v.bias, 0)

        def _adaLN_init(module):
            if isinstance(module, (AdaLNBlock, AdaLNFinalLayer)):
                # zero out adaLN modulation linear layers
                adaLN_mod_lin = cast(nn.Linear, module.adaLN_modulation[-1])
                nn.init.constant_(adaLN_mod_lin.weight, 0)
                nn.init.constant_(adaLN_mod_lin.bias, 0)

                if isinstance(module, AdaLNFinalLayer):
                    adaLN_final_lin = cast(nn.Linear, module.linear)
                    nn.init.constant_(adaLN_final_lin.weight, 0)
                    nn.init.constant_(adaLN_final_lin.weight, 0)

        self.apply(_basic_init)
        self.apply(_adaLN_init)
        self.apply(_qkv_mlp_init)

    def get_action(self, state: Tensor) -> Tensor:
        """
        Get action from the actor network.
        """
        B = state.size(0)
        actions = torch.zeros(B, self.act_dim, device=self.device).unsqueeze(-1)  # (B, act_dim, 1)
        actions = self.actor(state, actions)
        return actions
    

if __name__ == "__main__":
    gpt_config = Config()
    tf = Transformer(gpt_config, act_limit=0, use_adaln=True)
    
    torch.manual_seed(gpt_config.seed)
    bs = gpt_config.batch_size
    s_dim = gpt_config.state_dim
    a_dim = gpt_config.action_dim

    state = torch.randn(bs, s_dim, device=gpt_config.device).unsqueeze(-1)  # (B, s_dim, 1)

    actions = tf.get_action(state)

    print(actions)