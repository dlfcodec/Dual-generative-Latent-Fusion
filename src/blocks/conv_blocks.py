import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        dropout_prob: float = 0.0,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_channels_ = self.in_channels if self.out_channels is None else self.out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels_, kernel_size=kernel_size, padding=kernel_size//2, bias=False)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.out_channels_, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels_, kernel_size=kernel_size, padding=kernel_size//2, bias=False)

        if self.in_channels != self.out_channels_:
            self.nin_shortcut = nn.Conv2d(self.in_channels, self.out_channels_, kernel_size=1, bias=False)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels_:
            residual = self.nin_shortcut(hidden_states)

        return hidden_states + residual


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
    ):
        super().__init__()
        if out_channels == None:
            out_channels = in_channels

        self.layer_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, int(in_channels * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(in_channels * mlp_ratio), out_channels),
        )

        self.short = nn.Conv1d(in_channels, out_channels, kernel_size=1) if out_channels != in_channels else nn.Identity()

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape
        x = x * self.layer_scale.repeat(B, 1, H, W)
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = self.mlp(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        x = x + self.short(identity)
        return x