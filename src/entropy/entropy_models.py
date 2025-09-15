# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function


# pylint: disable=W0221
class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None
# pylint: enable=W0221


class EntropyCoder():
    def __init__(self, ec_thread=False, stream_part=1):
        super().__init__()

        from .MLCodec_rans import RansEncoder, RansDecoder
        self.encoder = RansEncoder(ec_thread, stream_part)
        self.decoder = RansDecoder(stream_part)

    @staticmethod
    def pmf_to_quantized_cdf(pmf, precision=16):
        from .MLCodec_CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
        cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
        cdf = torch.IntTensor(cdf)
        return cdf

    @staticmethod
    def pmf_to_cdf(pmf, tail_mass, pmf_length, max_length):
        entropy_coder_precision = 16
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = EntropyCoder.pmf_to_quantized_cdf(prob, entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def reset(self):
        self.encoder.reset()

    def add_cdf(self, cdf, cdf_length, offset):
        enc_cdf_idx = self.encoder.add_cdf(cdf, cdf_length, offset)
        dec_cdf_idx = self.decoder.add_cdf(cdf, cdf_length, offset)
        assert enc_cdf_idx == dec_cdf_idx
        return enc_cdf_idx

    def encode_with_indexes(self, symbols, indexes, cdf_group_index):
        self.encoder.encode_with_indexes(symbols.clamp(-30000, 30000).to(torch.int16).cpu().numpy(),
                                         indexes.to(torch.int16).cpu().numpy(),
                                         cdf_group_index)

    def encode_with_indexes_np(self, symbols, indexes, cdf_group_index):
        self.encoder.encode_with_indexes(symbols.clip(-30000, 30000).astype(np.int16).reshape(-1),
                                         indexes.astype(np.int16).reshape(-1),
                                         cdf_group_index)

    def flush(self):
        self.encoder.flush()

    def get_encoded_stream(self):
        return self.encoder.get_encoded_stream().tobytes()

    def set_stream(self, stream):
        self.decoder.set_stream((np.frombuffer(stream, dtype=np.uint8)))

    def decode_stream(self, indexes, cdf_group_index):
        rv = self.decoder.decode_stream(indexes.to(torch.int16).cpu().numpy(),
                                        cdf_group_index)
        rv = torch.Tensor(rv)
        return rv

    def decode_stream_np(self, indexes, cdf_group_index):
        rv = self.decoder.decode_stream(indexes.astype(np.int16).reshape(-1),
                                        cdf_group_index)
        return rv


class Bitparm(nn.Module):
    def __init__(self, channel, final=False):
        super().__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(
            torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(
            torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(
                torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        x = x * F.softplus(self.h) + self.b
        if self.final:
            return x

        return x + torch.tanh(x) * torch.tanh(self.a)


class AEHelper():
    def __init__(self):
        super().__init__()
        self.entropy_coder = None
        self.cdf_group_index = None
        self._offset = None
        self._quantized_cdf = None
        self._cdf_length = None

    def set_cdf_info(self, quantized_cdf, cdf_length, offset):
        self._quantized_cdf = quantized_cdf.cpu().numpy()
        self._cdf_length = cdf_length.reshape(-1).int().cpu().numpy()
        self._offset = offset.reshape(-1).int().cpu().numpy()

    def get_cdf_info(self):
        return self._quantized_cdf, \
            self._cdf_length, \
            self._offset


class BitEstimator(AEHelper, nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)
        self.channel = channel

    def forward(self, x):
        return self.get_cdf(x)

    def get_logits_cdf(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return x

    def get_cdf(self, x):
        return torch.sigmoid(self.get_logits_cdf(x))

    def get_prob(self, x):
        lower = self.get_logits_cdf(x - 0.5)
        upper = self.get_logits_cdf(x + 0.5)
        sign = -torch.sign(lower + upper)
        sign = sign.detach()
        prob = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower)
        )
        prob = LowerBound.apply(prob, 1e-9)
        return prob

    def update(self, force=False, entropy_coder=None):
        assert entropy_coder is not None
        self.entropy_coder = entropy_coder

        if not force and self._offset is not None:
            return

        with torch.no_grad():
            device = next(self.parameters()).device
            medians = torch.zeros((self.channel), device=device)

            minima = medians + 50
            for i in range(50, 1, -1):
                samples = torch.zeros_like(medians) - i
                samples = samples[None, :, None, None]
                probs = self.forward(samples)
                probs = torch.squeeze(probs)
                minima = torch.where(probs < torch.zeros_like(medians) + 0.0001,
                                     torch.zeros_like(medians) + i, minima)

            maxima = medians + 50
            for i in range(50, 1, -1):
                samples = torch.zeros_like(medians) + i
                samples = samples[None, :, None, None]
                probs = self.forward(samples)
                probs = torch.squeeze(probs)
                maxima = torch.where(probs > torch.zeros_like(medians) + 0.9999,
                                     torch.zeros_like(medians) + i, maxima)

            minima = minima.int()
            maxima = maxima.int()

            offset = -minima

            pmf_start = medians - minima
            pmf_length = maxima + minima + 1

            max_length = pmf_length.max()
            device = pmf_start.device
            samples = torch.arange(max_length, device=device)

            samples = samples[None, :] + pmf_start[:, None, None]

            half = float(0.5)

            lower = self.forward(samples - half).squeeze(0)
            upper = self.forward(samples + half).squeeze(0)
            pmf = upper - lower

            pmf = pmf[:, 0, :]
            tail_mass = lower[:, 0, :1] + (1.0 - upper[:, 0, -1:])

            quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
            cdf_length = pmf_length + 2
            self.set_cdf_info(quantized_cdf, cdf_length, offset)
            self.cdf_group_index = self.entropy_coder.add_cdf(*self.get_cdf_info())

    @staticmethod
    def build_indexes(size):
        N, C, H, W = size
        indexes = torch.arange(C, dtype=torch.int).view(1, -1, 1, 1)
        return indexes.repeat(N, 1, H, W)

    @staticmethod
    def build_indexes_np(size):
        return BitEstimator.build_indexes(size).cpu().numpy()

    def encode(self, x):
        indexes = self.build_indexes(x.size())
        return self.entropy_coder.encode_with_indexes(x.reshape(-1), indexes.reshape(-1),
                                                      self.cdf_group_index)

    def decode_stream(self, size, dtype, device):
        output_size = (1, self.channel, size[0], size[1])
        indexes = self.build_indexes(output_size)
        val = self.entropy_coder.decode_stream(indexes.reshape(-1), self.cdf_group_index)
        val = val.reshape(indexes.shape)
        return val.to(dtype).to(device)


class GaussianEncoder(AEHelper):
    def __init__(self, distribution='laplace'):
        super().__init__()
        assert distribution in ['laplace', 'gaussian']
        self.distribution = distribution
        if distribution == 'laplace':
            self.cdf_distribution = torch.distributions.laplace.Laplace
            self.scale_min = 0.01
            self.scale_max = 64.0
            self.scale_level = 256
        elif distribution == 'gaussian':
            self.cdf_distribution = torch.distributions.normal.Normal
            self.scale_min = 0.11
            self.scale_max = 64.0
            self.scale_level = 256
        self.scale_table = self.get_scale_table(self.scale_min, self.scale_max, self.scale_level)

        self.log_scale_min = math.log(self.scale_min)
        self.log_scale_max = math.log(self.scale_max)
        self.log_scale_step = (self.log_scale_max - self.log_scale_min) / (self.scale_level - 1)

    @staticmethod
    def get_scale_table(min_val, max_val, levels):
        return torch.exp(torch.linspace(math.log(min_val), math.log(max_val), levels))

    @staticmethod
    def get_gaussian_prob(values, scales):
        def _standardized_cumulative(inputs):
            half = float(0.5)
            const = float(-(2 ** -0.5))
            # Using the complementary error function maximizes numerical precision.
            return half * torch.erfc(const * inputs)

        def _cdf2(inputs):
            const = float(-(2 ** -0.5))
            return torch.erfc(const * inputs)

        scales = LowerBound.apply(scales, 0.11)
        values = torch.abs(values)
        upper = _cdf2((0.5 - values) / scales)
        lower = _cdf2((-0.5 - values) / scales)
        prob = upper - lower
        prob = LowerBound.apply(0.5 * prob, 1e-9)
        return prob

    @staticmethod
    def get_laplace_prob(values, scales):
        def _cdf(inputs):
            # this is the original function of cdf, but we only care diffence of cdf
            return 0.5 + 0.5 * torch.sign(inputs) * (1.0 - torch.exp(-torch.abs(inputs)))

        def _cdf2(inputs):
            return torch.sign(inputs) * (1.0 - torch.exp(-torch.abs(inputs)))

        scales = LowerBound.apply(scales, 0.01)
        upper = _cdf2((values + 0.5) / scales)
        lower = _cdf2((values - 0.5) / scales)
        prob = upper - lower
        prob = LowerBound.apply(0.5 * prob, 1e-9)
        return prob

    def update(self, force=False, entropy_coder=None):
        assert entropy_coder is not None
        self.entropy_coder = entropy_coder

        if not force and self._offset is not None:
            return

        pmf_center = torch.zeros_like(self.scale_table) + 50
        scales = torch.zeros_like(pmf_center) + self.scale_table
        mu = torch.zeros_like(scales)
        cdf_distribution = self.cdf_distribution(mu, scales)
        for i in range(50, 1, -1):
            samples = torch.zeros_like(pmf_center) + i
            probs = cdf_distribution.cdf(samples)
            probs = torch.squeeze(probs)
            pmf_center = torch.where(probs > torch.zeros_like(pmf_center) + 0.9999,
                                     torch.zeros_like(pmf_center) + i, pmf_center)

        pmf_center = pmf_center.int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.arange(max_length, device=device) - pmf_center[:, None]
        samples = samples.float()

        scales = torch.zeros_like(samples) + self.scale_table[:, None]
        mu = torch.zeros_like(scales)
        cdf_distribution = self.cdf_distribution(mu, scales)

        upper = cdf_distribution.cdf(samples + 0.5)
        lower = cdf_distribution.cdf(samples - 0.5)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)

        self.set_cdf_info(quantized_cdf, pmf_length+2, -pmf_center)
        self.cdf_group_index = self.entropy_coder.add_cdf(*self.get_cdf_info())

    def build_indexes(self, scales, skip_thres=None):
        scales = torch.maximum(scales, torch.zeros_like(scales) + 1e-5)
        indexes = (torch.log(scales) - self.log_scale_min) / self.log_scale_step
        indexes = indexes.clamp_(0, self.scale_level - 1)
        if skip_thres is not None:
            # indexes = torch.where(scales < 1e-4, torch.zeros_like(indexes) - 1, indexes)
            indexes = torch.where(scales < skip_thres, torch.zeros_like(indexes) - 1, indexes)
        return indexes.int()

    def encode(self, x, scales, skip_thres=None):
        indexes = self.build_indexes(scales, skip_thres)
        return self.entropy_coder.encode_with_indexes(x.reshape(-1), indexes.reshape(-1),
                                                      self.cdf_group_index)

    def decode_stream(self, scales, dtype, device, skip_thres=None):
        indexes = self.build_indexes(scales, skip_thres)
        val = self.entropy_coder.decode_stream(indexes.reshape(-1),
                                               self.cdf_group_index)
        val = val.reshape(scales.shape)
        return val.to(device).to(dtype)


###########
# Huffman #
###########

class Node:
    def __init__(self, index=None, left=None, right=None):
        self.index = index
        self.code = ""

        self.left = left
        self.right = right

        self.is_leaf = (index is not None)

    def add_code(self, code):
        if self.is_leaf:
            # is leaf
            self.code = code + self.code
        else:
            # is node
            self.left.add_code(code)
            self.right.add_code(code)

class HuffmanCodec_OneQP:

    def __init__(self, prob, device):
        self.device = device
        self.node = HuffmanCodec_OneQP.build_huffman_tree(prob)
        self.table = HuffmanCodec_OneQP.convert_huffman_tree_to_list(self.node)

    @staticmethod
    def build_huffman_tree(prob):
        node_list = [Node(i) for i in range(len(prob))]
        prob_list = list(prob)

        while len(node_list) > 1:
            min_index1, min_index2 = sorted(np.argpartition(prob_list, 1)[:2])

            node1 = node_list[min_index1]
            node1.add_code("0")
            node2 = node_list[min_index2]
            node2.add_code("1")
            node = Node(left=node1, right=node2)

            prob = prob_list[min_index1] + prob_list[min_index2]

            node_list = node_list[:min_index1] + node_list[min_index1 + 1:min_index2] + node_list[min_index2 + 1:] + [node]
            prob_list = prob_list[:min_index1] + prob_list[min_index1 + 1:min_index2] + prob_list[min_index2 + 1:] + [prob]
        
        return node_list[0]

    @staticmethod
    def convert_huffman_tree_to_list(node):
        huffman_dict = {}

        def search_tree(n):
            if n.is_leaf:
                huffman_dict[n.index] = n.code
            else:
                search_tree(n.left)
                search_tree(n.right)
        
        search_tree(node)
        huffman_table = [huffman_dict[i] for i in range(len(huffman_dict))]

        return huffman_table

    def compress(self, x):
        # x: 1, 1, 16, 16
        x = x.reshape(-1).detach().cpu().numpy()
        
        x_str = "1"
        for idx in x:
            x_str += self.table[idx]

        # Convert the binary string to an integer  
        x_byte = int(x_str, 2)  
        num_bytes = (x_byte.bit_length() + 7) // 8  
        bit_stream = x_byte.to_bytes(num_bytes, 'big')  
        return {"bit_stream" : bit_stream}

    def decompress(self, bit_stream):
        x_hat_byte = int.from_bytes(bit_stream, 'big')  
        x_hat_str = bin(x_hat_byte)[3:]

        x_hat_list = []

        idx = 0
        while idx < len(x_hat_str):
            this_node = self.node
            while not this_node.is_leaf:
                if x_hat_str[idx] == "0":
                    this_node = this_node.left
                else:
                    this_node = this_node.right
                idx += 1
            x_hat_list.append(this_node.index)

        return {"index" : torch.tensor(x_hat_list).to(self.device)}


class HuffmanCodec:

    def __init__(self):
        self.codec_list = {}

    def load_prob(self, prob_path, device):
        self.device = device
        probs = torch.load(prob_path)
        for qp in probs:
            self.codec_list[qp] = HuffmanCodec_OneQP(probs[qp].detach().cpu().numpy(), device)

    def compress(self, x, q_index):
        return self.codec_list[q_index].compress(x)

    def decompress(self, bit_stream, q_index): 
        return self.codec_list[q_index].decompress(bit_stream)