# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

import torch
from torch import nn

from pytorch_msssim import MS_SSIM

from .entropy_models import BitEstimator, GaussianEncoder, EntropyCoder, LowerBound, HuffmanCodec

def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    # padding_left = (new_w - width) // 2
    padding_left = 0
    padding_right = new_w - width - padding_left
    # padding_top = (new_h - height) // 2
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return padding_left, padding_right, padding_top, padding_bottom


class CompressionModel(nn.Module):
    def __init__(self, y_distribution, z_channel, mv_z_channel=None,
                 ec_thread=False, stream_part=1):
        super().__init__()

        self.y_distribution = y_distribution
        self.z_channel = z_channel
        self.mv_z_channel = mv_z_channel
        self.entropy_coder = None
        self.bit_estimator_z = None
        self.bit_estimator_z_mv = None
        if mv_z_channel is not None:
            self.bit_estimator_z_mv = BitEstimator(mv_z_channel)
        self.gaussian_encoder = GaussianEncoder(distribution=y_distribution)
        self.force_zero_thres = None
        self.noise_level = 0.5
        self.ec_thread = ec_thread
        self.stream_part = stream_part

        self.huffman_codec = HuffmanCodec()
        self.huffman_codec_flag = False

        self.mse = nn.MSELoss(reduction='none')
        self.ssim = MS_SSIM(data_range=1.0, size_average=False)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

        self.masks = {}
        self.force_generate_mask = False

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                torch.nn.init.xavier_normal_(m.weight, 1.)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
                
    def set_force_zero_thres(self, scale_thres):
        self.force_zero_thres = scale_thres

    def set_force_generate_mask(self, force):
        self.force_generate_mask = force

    def get_force_generate_mask(self):
        return self.force_generate_mask

    def get_z_cdf_info(self):
        return self.bit_estimator_z.get_cdf_info()

    def get_mv_z_cdf_info(self):
        assert self.bit_estimator_z_mv is not None
        return self.bit_estimator_z_mv.get_cdf_info()

    def get_y_cdf_info(self):
        return self.gaussian_encoder.get_cdf_info()

    def quant(self, x, force_detach=False):
        if self.training or force_detach:
            n = torch.round(x) - x
            n = n.clone().detach()
            return x + n

        return torch.round(x)

    def get_one_q(self, q, q_index):
        return q[q_index:q_index + 1, :, :, :]

    def get_curr_q(self, q, q_index, batch_size=1):
        if isinstance(q_index, list):
            q = [self.get_one_q(q, i) for i in q_index]
            q = torch.cat(q, dim=0)
            # print("q_scalce", q)
            q = q.repeat_interleave(batch_size, dim=0)
        else:
            q = self.get_one_q(q, q_index)

        return q

    @staticmethod
    def get_qp_num():
        return 4

    @staticmethod
    def get_anchor_num(q_index):
        if isinstance(q_index, list):
            return len(q_index)
        return 1
    
    def set_noise_level(self, noise_level):
        self.noise_level = noise_level

    def get_noise_level(self):
        return self.noise_level

    def add_noise(self, x):
        noise = torch.nn.init.uniform_(torch.zeros_like(x), -self.noise_level, self.noise_level)
        noise = noise.clone().detach()
        return x + noise

    @staticmethod
    def add_specified_noise(x, n):
        noise = torch.nn.init.uniform_(torch.zeros_like(x), -n, n)
        noise = noise.clone().detach()
        return x + noise

    @staticmethod
    def probs_to_bits(probs):
        bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
        bits = LowerBound.apply(bits, 0)
        return bits

    def get_y_gaussian_bits(self, y, sigma):
        if self.training:
            probs = GaussianEncoder.get_gaussian_prob(y, sigma)
        else:
            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-5, 1e10)
            gaussian = torch.distributions.normal.Normal(mu, sigma)
            probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return CompressionModel.probs_to_bits(probs)

    def get_y_laplace_bits(self, y, sigma):
        if self.training:
            probs = GaussianEncoder.get_laplace_prob(y, sigma)
        else:
            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-5, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return CompressionModel.probs_to_bits(probs)

    def get_z_bits(self, z, bit_estimator):
        if self.training:
            probs = bit_estimator.get_prob(z)
        else:
            probs = bit_estimator.get_cdf(z + 0.5) - bit_estimator.get_cdf(z - 0.5)
        return CompressionModel.probs_to_bits(probs)

    def update(self, force=False):
        self.entropy_coder = EntropyCoder(self.ec_thread, self.stream_part)
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)
    
    def update_vq(self, prob_path, device):
        self.huffman_codec.load_prob(prob_path, device)
        self.huffman_codec_flag = True

    def update_cdf(self, cdf):
        self.entropy_coder = EntropyCoder(self.ec_thread, self.stream_part)
        self.gaussian_encoder.set_entropy_coder(self.entropy_coder)
        self.gaussian_encoder.set_cdf_info(cdf['y']['quantized_cdf'],
                                           cdf['y']['cdf_length'], cdf['y']['offset'])

    def pad_for_y(self, y):
        if self.training:
            return y, None

        _, _, H, W = y.size()
        padding_l, padding_r, padding_t, padding_b = get_padding_size(H, W, 4)
        y_pad = torch.nn.functional.pad(
            y,
            (padding_l, padding_r, padding_t, padding_b),
            mode="replicate",
        )
        return y_pad, (-padding_l, -padding_r, -padding_t, -padding_b)

    @staticmethod
    def get_to_y_slice_shape(height, width):
        padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, 4)
        return (-padding_l, -padding_r, -padding_t, -padding_b)

    def slice_to_y(self, param, slice_shape):
        if self.training:
            return param

        return torch.nn.functional.pad(param, slice_shape)


    @staticmethod
    def separate_prior(params):
        return params.chunk(3, 1)

    def get_mask(self, height, width, dtype, device):
        curr_mask_str = f"{width}x{height}"
        if curr_mask_str not in self.masks or self.force_generate_mask:
            micro_mask = torch.tensor(((1, 0), (0, 1)), dtype=dtype, device=device)
            mask_0 = micro_mask.repeat((height + 1) // 2, (width + 1) // 2)
            mask_0 = mask_0[:height, :width]
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_1 = torch.ones_like(mask_0) - mask_0
            self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]

    def process_with_mask(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = self.quant(y_res)
        
        if not self.training and self.force_zero_thres is not None:
            # y_q[:, self.force_zero_thres] = 0.
            # scales_hat[:, self.force_zero_thres] = 0.
            cond = scales_hat < self.force_zero_thres
            y_q = torch.where(cond, torch.zeros_like(y_q), y_q)
            scales_hat = torch.where(cond, torch.zeros_like(scales_hat), scales_hat)
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat

    @staticmethod
    def get_one_channel_four_parts_mask(height, width, dtype, device):
        micro_mask_0 = torch.tensor(((1, 0), (0, 0)), dtype=dtype, device=device)
        mask_0 = micro_mask_0.repeat((height + 1) // 2, (width + 1) // 2)
        mask_0 = mask_0[:height, :width]
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_0 = torch.unsqueeze(mask_0, 0)

        micro_mask_1 = torch.tensor(((0, 1), (0, 0)), dtype=dtype, device=device)
        mask_1 = micro_mask_1.repeat((height + 1) // 2, (width + 1) // 2)
        mask_1 = mask_1[:height, :width]
        mask_1 = torch.unsqueeze(mask_1, 0)
        mask_1 = torch.unsqueeze(mask_1, 0)

        micro_mask_2 = torch.tensor(((0, 0), (1, 0)), dtype=dtype, device=device)
        mask_2 = micro_mask_2.repeat((height + 1) // 2, (width + 1) // 2)
        mask_2 = mask_2[:height, :width]
        mask_2 = torch.unsqueeze(mask_2, 0)
        mask_2 = torch.unsqueeze(mask_2, 0)

        micro_mask_3 = torch.tensor(((0, 0), (0, 1)), dtype=dtype, device=device)
        mask_3 = micro_mask_3.repeat((height + 1) // 2, (width + 1) // 2)
        mask_3 = mask_3[:height, :width]
        mask_3 = torch.unsqueeze(mask_3, 0)
        mask_3 = torch.unsqueeze(mask_3, 0)

        return mask_0, mask_1, mask_2, mask_3

    def get_mask_four_parts(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        with torch.no_grad():
            if curr_mask_str not in self.masks or self.force_generate_mask:
                assert channel % 4 == 0
                m = torch.ones((batch, channel // 4, height, width), dtype=dtype, device=device)
                m0, m1, m2, m3 = self.get_one_channel_four_parts_mask(height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
                mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
                mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
                mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    @staticmethod
    def combine_four_parts(x_0_0, x_0_1, x_0_2, x_0_3,
                           x_1_0, x_1_1, x_1_2, x_1_3,
                           x_2_0, x_2_1, x_2_2, x_2_3,
                           x_3_0, x_3_1, x_3_2, x_3_3):
        x_0 = x_0_0 + x_0_1 + x_0_2 + x_0_3
        x_1 = x_1_0 + x_1_1 + x_1_2 + x_1_3
        x_2 = x_2_0 + x_2_1 + x_2_2 + x_2_3
        x_3 = x_3_0 + x_3_1 + x_3_2 + x_3_3
        return torch.cat((x_0, x_1, x_2, x_3), dim=1)

    @staticmethod
    def combine_for_writing(x, force_zero_thres=None):
        # if force_zero_thres is not None:
        #     x[:, force_zero_thres] = 0.
        x0, x1, x2, x3 = x.chunk(4, 1)
        return (x0 + x1) + (x2 + x3)

    def forward_four_part_prior(self, y, common_params,
                                y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                y_spatial_prior_adaptor_3, y_spatial_prior,
                                y_spatial_prior_reduction=None, write=False):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        quant_step, scales, means = self.separate_prior(common_params)
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        if self.training:
            quant_step = LowerBound.apply(quant_step, 0.5)
        else:
            quant_step = torch.clamp_min(quant_step, 0.5)
        y = y / quant_step

        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        y_res_2, y_q_2, y_hat_2, s_hat_2 = self.process_with_mask(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        y_res_3, y_q_3, y_hat_3, s_hat_3 = self.process_with_mask(y, scales, means, mask_3)

        y_res = (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
        y_q = (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
        y_hat = y_hat_so_far + y_hat_3
        scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)

        y_hat = y_hat * quant_step

        if write:
            y_q_w_0 = self.combine_for_writing(y_q_0)
            y_q_w_1 = self.combine_for_writing(y_q_1)
            y_q_w_2 = self.combine_for_writing(y_q_2)
            y_q_w_3 = self.combine_for_writing(y_q_3)
            scales_w_0 = self.combine_for_writing(s_hat_0)
            scales_w_1 = self.combine_for_writing(s_hat_1)
            scales_w_2 = self.combine_for_writing(s_hat_2)
            scales_w_3 = self.combine_for_writing(s_hat_3)
            return y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
                scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat
        return y_res, y_q, y_hat, scales_hat

    def compress_four_part_prior(self, y, common_params,
                                 y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                 y_spatial_prior_adaptor_3, y_spatial_prior,
                                 y_spatial_prior_reduction=None):
        return self.forward_four_part_prior(y, common_params,
                                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                            y_spatial_prior_adaptor_3, y_spatial_prior,
                                            y_spatial_prior_reduction, write=True)

    def decompress_four_part_prior(self, common_params,
                                   y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                   y_spatial_prior_adaptor_3, y_spatial_prior,
                                   y_spatial_prior_reduction=None):
        quant_step, scales, means = self.separate_prior(common_params)
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)
        quant_step = torch.clamp_min(quant_step, 0.5)

        scales_r = self.combine_for_writing(scales * mask_0, self.force_zero_thres)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device, self.force_zero_thres)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_0
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        scales_r = self.combine_for_writing(scales * mask_1, self.force_zero_thres)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device, self.force_zero_thres)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_1
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        scales_r = self.combine_for_writing(scales * mask_2, self.force_zero_thres)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device, self.force_zero_thres)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_2
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        scales_r = self.combine_for_writing(scales * mask_3, self.force_zero_thres)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device, self.force_zero_thres)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_3
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        y_hat = y_hat_so_far * quant_step

        return y_hat


    def process_with_mask_recon_with_z(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = self.quant(y_res)
        y_hat = y_q * 0. + means_hat

        return y_hat, means_hat


    def forward_four_part_prior_recon_with_z(self, y, common_params,
                                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                            y_spatial_prior_adaptor_3, y_spatial_prior,
                                            y_spatial_prior_reduction=None, write=False):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        quant_step, scales, means = self.separate_prior(common_params)
        # print("q", quant_step.abs().max(), quant_step.abs().mean())
        # print("scale", scales.abs().max(), scales.abs().mean())
        # print("means", means.abs().max(), means.abs().mean())
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        if self.training:
            quant_step = LowerBound.apply(quant_step, 0.5)
        else:
            quant_step = torch.clamp_min(quant_step, 0.5)
        y = y / quant_step

        y_hat_0, m_hat_0 = self.process_with_mask_recon_with_z(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        y_hat_1, m_hat_1 = self.process_with_mask_recon_with_z(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        y_hat_2, m_hat_2 = self.process_with_mask_recon_with_z(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        y_hat_3, m_hat_3 = self.process_with_mask_recon_with_z(y, scales, means, mask_3)

        y_hat = y_hat_so_far + y_hat_3
        # y_hat = m_hat_0 + m_hat_1 + m_hat_2 + m_hat_3

        y_hat = y_hat * quant_step

        return  y_hat



class CompressionModel_type2(nn.Module):
    """
    Remove the adaptive quant design.
    """
    def __init__(self, y_distribution, z_channel, mv_z_channel=None,
                 ec_thread=False, stream_part=1):
        super().__init__()

        self.y_distribution = y_distribution
        self.z_channel = z_channel
        self.mv_z_channel = mv_z_channel
        self.entropy_coder = None
        self.bit_estimator_z = None
        self.bit_estimator_z_mv = None
        if mv_z_channel is not None:
            self.bit_estimator_z_mv = BitEstimator(mv_z_channel)
        self.gaussian_encoder = GaussianEncoder(distribution=y_distribution)
        self.force_zero_thres = None
        self.noise_level = 0.5
        self.ec_thread = ec_thread
        self.stream_part = stream_part

        self.huffman_codec = HuffmanCodec()
        self.huffman_codec_flag = False

        self.mse = nn.MSELoss(reduction='none')
        self.ssim = MS_SSIM(data_range=1.0, size_average=False)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

        self.masks = {}
        self.force_generate_mask = False

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                torch.nn.init.xavier_normal_(m.weight, 1.)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
                
    def set_force_zero_thres(self, scale_thres):
        self.force_zero_thres = scale_thres

    def set_force_generate_mask(self, force):
        self.force_generate_mask = force

    def get_force_generate_mask(self):
        return self.force_generate_mask

    def get_z_cdf_info(self):
        return self.bit_estimator_z.get_cdf_info()

    def get_mv_z_cdf_info(self):
        assert self.bit_estimator_z_mv is not None
        return self.bit_estimator_z_mv.get_cdf_info()

    def get_y_cdf_info(self):
        return self.gaussian_encoder.get_cdf_info()

    def quant(self, x, force_detach=False):
        if self.training or force_detach:
            n = torch.round(x) - x
            n = n.clone().detach()
            return x + n

        return torch.round(x)

    def get_one_q(self, q, q_index):
        return q[q_index:q_index + 1, :, :, :]

    def get_curr_q(self, q, q_index, batch_size=1):
        if isinstance(q_index, list):
            q = [self.get_one_q(q, i) for i in q_index]
            q = torch.cat(q, dim=0)
            # print("q_scalce", q)
            q = q.repeat_interleave(batch_size, dim=0)
        else:
            q = self.get_one_q(q, q_index)

        return q

    @staticmethod
    def get_qp_num():
        return 4

    @staticmethod
    def get_anchor_num(q_index):
        if isinstance(q_index, list):
            return len(q_index)
        return 1
    
    def set_noise_level(self, noise_level):
        self.noise_level = noise_level

    def get_noise_level(self):
        return self.noise_level

    def add_noise(self, x):
        noise = torch.nn.init.uniform_(torch.zeros_like(x), -self.noise_level, self.noise_level)
        noise = noise.clone().detach()
        return x + noise

    @staticmethod
    def add_specified_noise(x, n):
        noise = torch.nn.init.uniform_(torch.zeros_like(x), -n, n)
        noise = noise.clone().detach()
        return x + noise

    @staticmethod
    def probs_to_bits(probs):
        bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
        bits = LowerBound.apply(bits, 0)
        return bits

    def get_y_gaussian_bits(self, y, sigma):
        if self.training:
            probs = GaussianEncoder.get_gaussian_prob(y, sigma)
        else:
            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-5, 1e10)
            gaussian = torch.distributions.normal.Normal(mu, sigma)
            probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return CompressionModel.probs_to_bits(probs)

    def get_y_laplace_bits(self, y, sigma):
        if self.training:
            probs = GaussianEncoder.get_laplace_prob(y, sigma)
        else:
            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-5, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return CompressionModel.probs_to_bits(probs)

    def get_z_bits(self, z, bit_estimator):
        if self.training:
            probs = bit_estimator.get_prob(z)
        else:
            probs = bit_estimator.get_cdf(z + 0.5) - bit_estimator.get_cdf(z - 0.5)
        return CompressionModel.probs_to_bits(probs)

    def update(self, force=False):
        self.entropy_coder = EntropyCoder(self.ec_thread, self.stream_part)
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)
    
    def update_vq(self, prob_path, device):
        self.huffman_codec.load_prob(prob_path, device)
        self.huffman_codec_flag = True

    def update_cdf(self, cdf):
        self.entropy_coder = EntropyCoder(self.ec_thread, self.stream_part)
        self.gaussian_encoder.set_entropy_coder(self.entropy_coder)
        self.gaussian_encoder.set_cdf_info(cdf['y']['quantized_cdf'],
                                           cdf['y']['cdf_length'], cdf['y']['offset'])

    def pad_for_y(self, y):
        if self.training:
            return y, None

        _, _, H, W = y.size()
        padding_l, padding_r, padding_t, padding_b = get_padding_size(H, W, 4)
        y_pad = torch.nn.functional.pad(
            y,
            (padding_l, padding_r, padding_t, padding_b),
            mode="replicate",
        )
        return y_pad, (-padding_l, -padding_r, -padding_t, -padding_b)

    @staticmethod
    def get_to_y_slice_shape(height, width):
        padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, 4)
        return (-padding_l, -padding_r, -padding_t, -padding_b)

    def slice_to_y(self, param, slice_shape):
        if self.training:
            return param

        return torch.nn.functional.pad(param, slice_shape)


    @staticmethod
    def separate_prior(params):
        return params.chunk(2, 1)

    def get_mask(self, height, width, dtype, device):
        curr_mask_str = f"{width}x{height}"
        if curr_mask_str not in self.masks or self.force_generate_mask:
            micro_mask = torch.tensor(((1, 0), (0, 1)), dtype=dtype, device=device)
            mask_0 = micro_mask.repeat((height + 1) // 2, (width + 1) // 2)
            mask_0 = mask_0[:height, :width]
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_1 = torch.ones_like(mask_0) - mask_0
            self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]

    def process_with_mask(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = self.quant(y_res)
        
        if not self.training and self.force_zero_thres is not None:
            # y_q[:, self.force_zero_thres] = 0.
            # scales_hat[:, self.force_zero_thres] = 0.
            cond = scales_hat < self.force_zero_thres
            y_q = torch.where(cond, torch.zeros_like(y_q), y_q)
            scales_hat = torch.where(cond, torch.zeros_like(scales_hat), scales_hat)
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat

    @staticmethod
    def get_one_channel_four_parts_mask(height, width, dtype, device):
        micro_mask_0 = torch.tensor(((1, 0), (0, 0)), dtype=dtype, device=device)
        mask_0 = micro_mask_0.repeat((height + 1) // 2, (width + 1) // 2)
        mask_0 = mask_0[:height, :width]
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_0 = torch.unsqueeze(mask_0, 0)

        micro_mask_1 = torch.tensor(((0, 1), (0, 0)), dtype=dtype, device=device)
        mask_1 = micro_mask_1.repeat((height + 1) // 2, (width + 1) // 2)
        mask_1 = mask_1[:height, :width]
        mask_1 = torch.unsqueeze(mask_1, 0)
        mask_1 = torch.unsqueeze(mask_1, 0)

        micro_mask_2 = torch.tensor(((0, 0), (1, 0)), dtype=dtype, device=device)
        mask_2 = micro_mask_2.repeat((height + 1) // 2, (width + 1) // 2)
        mask_2 = mask_2[:height, :width]
        mask_2 = torch.unsqueeze(mask_2, 0)
        mask_2 = torch.unsqueeze(mask_2, 0)

        micro_mask_3 = torch.tensor(((0, 0), (0, 1)), dtype=dtype, device=device)
        mask_3 = micro_mask_3.repeat((height + 1) // 2, (width + 1) // 2)
        mask_3 = mask_3[:height, :width]
        mask_3 = torch.unsqueeze(mask_3, 0)
        mask_3 = torch.unsqueeze(mask_3, 0)

        return mask_0, mask_1, mask_2, mask_3

    def get_mask_four_parts(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        with torch.no_grad():
            if curr_mask_str not in self.masks or self.force_generate_mask:
                assert channel % 4 == 0
                m = torch.ones((batch, channel // 4, height, width), dtype=dtype, device=device)
                m0, m1, m2, m3 = self.get_one_channel_four_parts_mask(height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
                mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
                mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
                mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    @staticmethod
    def combine_four_parts(x_0_0, x_0_1, x_0_2, x_0_3,
                           x_1_0, x_1_1, x_1_2, x_1_3,
                           x_2_0, x_2_1, x_2_2, x_2_3,
                           x_3_0, x_3_1, x_3_2, x_3_3):
        x_0 = x_0_0 + x_0_1 + x_0_2 + x_0_3
        x_1 = x_1_0 + x_1_1 + x_1_2 + x_1_3
        x_2 = x_2_0 + x_2_1 + x_2_2 + x_2_3
        x_3 = x_3_0 + x_3_1 + x_3_2 + x_3_3
        return torch.cat((x_0, x_1, x_2, x_3), dim=1)

    @staticmethod
    def combine_for_writing(x, force_zero_thres=None):
        # if force_zero_thres is not None:
        #     x[:, force_zero_thres] = 0.
        x0, x1, x2, x3 = x.chunk(4, 1)
        return (x0 + x1) + (x2 + x3)

    def forward_four_part_prior(self, y, common_params,
                                y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                y_spatial_prior_adaptor_3, y_spatial_prior,
                                y_spatial_prior_reduction=None, write=False):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        scales, means = self.separate_prior(common_params)
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        y_res_2, y_q_2, y_hat_2, s_hat_2 = self.process_with_mask(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        y_res_3, y_q_3, y_hat_3, s_hat_3 = self.process_with_mask(y, scales, means, mask_3)

        y_res = (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
        y_q = (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
        y_hat = y_hat_so_far + y_hat_3
        scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)

        if write:
            y_q_w_0 = self.combine_for_writing(y_q_0)
            y_q_w_1 = self.combine_for_writing(y_q_1)
            y_q_w_2 = self.combine_for_writing(y_q_2)
            y_q_w_3 = self.combine_for_writing(y_q_3)
            scales_w_0 = self.combine_for_writing(s_hat_0)
            scales_w_1 = self.combine_for_writing(s_hat_1)
            scales_w_2 = self.combine_for_writing(s_hat_2)
            scales_w_3 = self.combine_for_writing(s_hat_3)
            return y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
                scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat
        return y_res, y_q, y_hat, scales_hat

    def compress_four_part_prior(self, y, common_params,
                                 y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                 y_spatial_prior_adaptor_3, y_spatial_prior,
                                 y_spatial_prior_reduction=None):
        return self.forward_four_part_prior(y, common_params,
                                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                            y_spatial_prior_adaptor_3, y_spatial_prior,
                                            y_spatial_prior_reduction, write=True)

    def decompress_four_part_prior(self, common_params,
                                   y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                   y_spatial_prior_adaptor_3, y_spatial_prior,
                                   y_spatial_prior_reduction=None):
        scales, means = self.separate_prior(common_params)
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        scales_r = self.combine_for_writing(scales * mask_0, self.force_zero_thres)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device, self.force_zero_thres)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_0
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        scales_r = self.combine_for_writing(scales * mask_1, self.force_zero_thres)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device, self.force_zero_thres)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_1
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        scales_r = self.combine_for_writing(scales * mask_2, self.force_zero_thres)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device, self.force_zero_thres)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_2
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        scales_r = self.combine_for_writing(scales * mask_3, self.force_zero_thres)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device, self.force_zero_thres)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_3
        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        y_hat = y_hat_so_far

        return y_hat


    def process_with_mask_recon_with_z(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = self.quant(y_res)
        y_hat = y_q * 0. + means_hat

        return y_hat, means_hat


    def forward_four_part_prior_recon_with_z(self, y, common_params,
                                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                            y_spatial_prior_adaptor_3, y_spatial_prior,
                                            y_spatial_prior_reduction=None, write=False):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        scales, means = self.separate_prior(common_params)
        # print("q", quant_step.abs().max(), quant_step.abs().mean())
        # print("scale", scales.abs().max(), scales.abs().mean())
        # print("means", means.abs().max(), means.abs().mean())
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        y_hat_0, m_hat_0 = self.process_with_mask_recon_with_z(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        y_hat_1, m_hat_1 = self.process_with_mask_recon_with_z(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        y_hat_2, m_hat_2 = self.process_with_mask_recon_with_z(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        y_hat_3, m_hat_3 = self.process_with_mask_recon_with_z(y, scales, means, mask_3)

        y_hat = y_hat_so_far + y_hat_3
        # y_hat = m_hat_0 + m_hat_1 + m_hat_2 + m_hat_3

        return  y_hat
    

class CompressionModel_type3(CompressionModel):
    def __init__(self, y_distribution, z_channel, mv_z_channel=None, ec_thread=False, stream_part=1):
        """
        Same as type1, but ignore the adaptive quant scalar.
        """
        super().__init__(y_distribution, z_channel, mv_z_channel, ec_thread, stream_part)
    
    def forward_four_part_prior(self, y, common_params,
                                y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                y_spatial_prior_adaptor_3, y_spatial_prior,
                                y_spatial_prior_reduction=None, write=False):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        quant_step, scales, means = self.separate_prior(common_params)
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        y_res_2, y_q_2, y_hat_2, s_hat_2 = self.process_with_mask(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        y_res_3, y_q_3, y_hat_3, s_hat_3 = self.process_with_mask(y, scales, means, mask_3)

        y_res = (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
        y_q = (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
        y_hat = y_hat_so_far + y_hat_3
        scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)

        if write:
            y_q_w_0 = self.combine_for_writing(y_q_0)
            y_q_w_1 = self.combine_for_writing(y_q_1)
            y_q_w_2 = self.combine_for_writing(y_q_2)
            y_q_w_3 = self.combine_for_writing(y_q_3)
            scales_w_0 = self.combine_for_writing(s_hat_0)
            scales_w_1 = self.combine_for_writing(s_hat_1)
            scales_w_2 = self.combine_for_writing(s_hat_2)
            scales_w_3 = self.combine_for_writing(s_hat_3)
            return y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
                scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat
        return y_res, y_q, y_hat, scales_hat

    def compress_four_part_prior(self, y, common_params,
                                 y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                 y_spatial_prior_adaptor_3, y_spatial_prior,
                                 y_spatial_prior_reduction=None):
        return self.forward_four_part_prior(y, common_params,
                                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                            y_spatial_prior_adaptor_3, y_spatial_prior,
                                            y_spatial_prior_reduction, write=True)

    def decompress_four_part_prior(self, common_params,
                                   y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                   y_spatial_prior_adaptor_3, y_spatial_prior,
                                   y_spatial_prior_reduction=None):
        quant_step, scales, means = self.separate_prior(common_params)
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        scales_r = self.combine_for_writing(scales * mask_0, self.force_zero_thres)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device, self.force_zero_thres)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_0
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        scales_r = self.combine_for_writing(scales * mask_1, self.force_zero_thres)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device, self.force_zero_thres)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_1
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        scales_r = self.combine_for_writing(scales * mask_2, self.force_zero_thres)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device, self.force_zero_thres)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_2
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        scales_r = self.combine_for_writing(scales * mask_3, self.force_zero_thres)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device, self.force_zero_thres)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_3
        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        y_hat = y_hat_so_far

        return y_hat


    def forward_four_part_prior_recon_with_z(self, y, common_params,
                                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                            y_spatial_prior_adaptor_3, y_spatial_prior,
                                            y_spatial_prior_reduction=None, write=False):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        quant_step, scales, means = self.separate_prior(common_params)
        # print("q", quant_step.abs().max(), quant_step.abs().mean())
        # print("scale", scales.abs().max(), scales.abs().mean())
        # print("means", means.abs().max(), means.abs().mean())
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        y_hat_0, m_hat_0 = self.process_with_mask_recon_with_z(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        y_hat_1, m_hat_1 = self.process_with_mask_recon_with_z(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        y_hat_2, m_hat_2 = self.process_with_mask_recon_with_z(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        y_hat_3, m_hat_3 = self.process_with_mask_recon_with_z(y, scales, means, mask_3)

        y_hat = y_hat_so_far + y_hat_3

        return  y_hat
