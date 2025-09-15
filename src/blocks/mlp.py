import numpy as np
import random as rd
import torch
from torch import nn
from einops import rearrange

class conv_mlp(nn.Module):
    def __init__(self, in_features: int, out_features: int, mlp_ratio=4.0, 
                 short_cut=False, norm_layer=nn.LayerNorm, act_func=nn.GELU) -> None:
        super().__init__()
        mid_features = int(in_features * mlp_ratio)
        self.feat_in = nn.Linear(in_features, mid_features)
        self.norm = norm_layer(mid_features)
        self.act = act_func()
        self.feat_out = nn.Linear(mid_features, out_features)
        self.short_cut = short_cut
        if short_cut:
            self.short = nn.Linear(in_features, out_features) \
                if in_features != out_features else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        identity = x
        x = self.feat_in(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.feat_out(x)
        if self.short_cut:
            x = x + self.short(identity)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x