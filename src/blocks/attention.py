import torch
from collections import OrderedDict
from torch import nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, mlp_ratio = 4.0, act_layer = nn.GELU, norm_layer = nn.LayerNorm):
        super().__init__()
        self.ln_1_x = norm_layer(d_model)
        self.ln_1_c = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(self, x: torch.Tensor, c: torch.Tensor):
        return self.attn(x, c, c, need_weights=False)[0]

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        attn_output = self.attention(x=self.ln_1_x(x), c=self.ln_1_c(c))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x