import torch
from collections import OrderedDict
from torch import nn


def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)