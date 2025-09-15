import torch
from torch import nn
from collections import OrderedDict

from titok.blocks import ResidualAttentionBlock
from einops import rearrange
from blocks.zero_conv import zero_Linear


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, d_cond, n_head, mlp_ratio = 4.0, act_layer = nn.GELU, norm_layer = nn.LayerNorm):
        super().__init__()

        self.ln_1_feat = norm_layer(d_model)
        self.ln_1_cond = norm_layer(d_cond)
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=d_cond, vdim=d_cond)
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

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        attn_output = self.attention(x=self.ln_1_feat(x), c=self.ln_1_cond(c))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x


class Interactive_crossAttn_type4(nn.Module):
    """ Perform attention with one transformer.
    """
    def __init__(self, titok_width, feat_width, num_attns=3, feat_patch_size=8, titok_patch_size=16, extra_titok_tokens=32+1, mlp_ratio=4.0) -> None:
        super().__init__()
        self.titok_width = titok_width
        self.feat_width = feat_width
        self.feat_patch_size = feat_patch_size
        self.titok_patch_size = titok_patch_size
        self.extra_titok_tokens = extra_titok_tokens

        # pos_emb
        self.titok_pos_emb = nn.Parameter(torch.zeros((titok_patch_size**2+extra_titok_tokens, 1, titok_width)), requires_grad=True)
        self.feat_pos_emb = nn.Parameter(torch.zeros((feat_patch_size**2, 1, feat_width)), requires_grad=True)

        # attn
        self.titok_compress_proj = nn.Linear(titok_width, feat_width)
        self.attn = nn.Sequential(*[
            ResidualAttentionBlock(feat_width, feat_width//64, mlp_ratio=mlp_ratio)
            for i in range(num_attns)
        ])
        self.titok_decompress_proj = nn.Sequential(
            nn.Linear(feat_width, feat_width*2),
            nn.LayerNorm(feat_width*2),
            nn.SiLU(),
        )

        # zero module
        self.feat_add = nn.Sequential(nn.LayerNorm(feat_width), nn.Linear(feat_width, feat_width))
        self.zero_add = zero_Linear(feat_width*2, titok_width)


    def set_to_zero(self):
        self.zero_add.set_to_zero()


    def forward(self, f_feat: torch.Tensor, f_titok_stack: torch.Tensor, stack_shape):
        nH, nW = stack_shape
        # rearrange feat stack
        f_feat_stack = rearrange(f_feat, 'b c (nH h) (nW w) -> (h w) (b nH nW) c', 
                                h=self.feat_patch_size, w=self.feat_patch_size, nH=nH, nW=nW).contiguous()

        # add positional encoding
        f_feat_stack_pos = self.feat_pos_emb.repeat(1, f_feat_stack.shape[1], 1) + f_feat_stack
        f_titok_stack_pos = self.titok_pos_emb.repeat(1, f_titok_stack.shape[1], 1) + f_titok_stack
        f_titok_stack_pos = self.titok_compress_proj(f_titok_stack_pos)

        # unify attn
        f = self.attn(torch.cat([f_titok_stack_pos, f_feat_stack_pos], dim=0))
        f_feat_stack_new = f[-(self.feat_patch_size**2):, :, :]
        f_titok_stack_new = f[:-(self.feat_patch_size**2), :, :]
        
        # add to original, feat
        f_feat_stack = f_feat_stack + self.feat_add(f_feat_stack_new)
        f_titok_stack = f_titok_stack + self.zero_add(self.titok_decompress_proj(f_titok_stack_new))
        
        # rearrange feat stack back
        f_feat = rearrange(f_feat_stack, '(h w) (b nH nW) c -> b c (nH h) (nW w)', 
                                h=self.feat_patch_size, w=self.feat_patch_size, nH=nH, nW=nW).contiguous()
        return f_feat, f_titok_stack

