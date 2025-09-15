import numpy as np
import random as rd
import torch
from torch import nn

class zero_Conv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 padding_mode='zeros', device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        for p in self.parameters():
            p.detach().zero_()
    
    def set_to_zero(self):
        for p in self.parameters():
            p.detach().zero_()


class zero_Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        for p in self.parameters():
            p.detach().zero_()

    def set_to_zero(self):
        for p in self.parameters():
            p.detach().zero_()