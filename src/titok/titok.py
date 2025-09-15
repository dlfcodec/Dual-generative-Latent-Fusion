"""This file contains the model definition of TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""

import torch
import torch.nn as nn
from einops import rearrange

from .blocks import TiTokEncoder, TiTokDecoder
from .quantizer import VectorQuantizer
from .maskgit_vqgan import Encoder as Pixel_Encoder
from .maskgit_vqgan import Decoder as Pixel_Decoder
from .maskgit_vqgan import VectorQuantizer as Pixel_Quantizer
from omegaconf import OmegaConf


class PretrainedTokenizer(nn.Module):
    def __init__(self, pretrained_weight_path=None):
        super().__init__()
        conf = OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
            "num_resolutions": 5,
            "dropout": 0.0,
            "hidden_channels": 128,
            "num_channels": 3,
            "num_res_blocks": 2,
            "resolution": 256,
            "z_channels": 256})
        self.encoder = Pixel_Encoder(conf)
        self.decoder = Pixel_Decoder(conf)
        self.quantize = Pixel_Quantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
        # Load pretrained weights
        if pretrained_weight_path:
            self.load_ckpt(pretrained_weight_path)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def load_ckpt(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu")), strict=True)
    
    @torch.no_grad()
    def encode(self, x):
        hidden_states = self.encoder(x)
        quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states)
        return quantized_states, codebook_indices.detach()

    def decode(self, quantized_states):
        rec_images = self.decoder(quantized_states)
        return rec_images
    
    @torch.no_grad()
    def decode_from_indices(self, codes):
        quantized_states = self.quantize.get_codebook_entry(codes)
        return self.decode(quantized_states)


class TiTok(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = TiTokEncoder(config)
        self.decoder = TiTokDecoder(config)
        
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)

        self.quantize = VectorQuantizer(
            codebook_size=config.model.vq_model.codebook_size,
            token_size=config.model.vq_model.token_size,
            commitment_cost=config.model.vq_model.commitment_cost,
            use_l2_norm=config.model.vq_model.use_l2_norm,)
        
        self.pixel_quantize = Pixel_Quantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
        self.pixel_decoder = Pixel_Decoder(OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
             "num_resolutions": 5,
             "dropout": 0.0,
             "hidden_channels": 128,
             "num_channels": 3,
             "num_res_blocks": 2,
             "resolution": 256,
             "z_channels": 256}))

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        z_quantized, result_dict = self.quantize(z)
        return z_quantized, result_dict
    
    def decode(self, z_quantized):
        decoded_latent = self.decoder(z_quantized)
        quantized_states = torch.einsum(
            'nchw,cd->ndhw', decoded_latent.softmax(1),
            self.pixel_quantize.embedding.weight)
        decoded = self.pixel_decoder(quantized_states)
        return decoded
    
    def decode_tokens(self, tokens):
        tokens = tokens.squeeze(1)
        batch, seq_len = tokens.shape # B x N
        z_quantized = self.quantize.get_codebook_entry(
            tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
        if self.quantize.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        decoded = self.decode(z_quantized)
        return decoded

    # added by xnf
    def forward_to_latent(self, x):
        z_quantized, result_dict = self.encode(x)
        decoded_latent = self.decoder(z_quantized)
        quantized_states = torch.einsum(
            'nchw,cd->ndhw', decoded_latent.softmax(1),
            self.pixel_quantize.embedding.weight)
        return quantized_states

    # added by xnf
    @torch.no_grad()
    def decode_tokens_to_latent(self, tokens):
        tokens = tokens.squeeze(1)
        batch, seq_len = tokens.shape # B x N
        z_quantized = self.quantize.get_codebook_entry(
            tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
        if self.quantize.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        decoded_latent = self.decoder(z_quantized)
        return decoded_latent
    
    # added by xnf
    @torch.no_grad()
    def decode_vqgan_latent(self, decoded_latent):
        quantized_states = torch.einsum(
            'nchw,cd->ndhw', decoded_latent.softmax(1),
            self.pixel_quantize.embedding.weight)
        decoded = self.pixel_decoder(quantized_states)
        return decoded
    
    # added by xnf
    def make_img_stack(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert H % 256 == 0 and W % 256 == 0
        nH, nW = H // 256, W // 256
        x_stack = x.unfold(2, 256, 256).unfold(3, 256, 256)
        x_stack = x_stack.reshape(B, C, -1, 256, 256)
        x_stack = x_stack.permute(0, 2, 1, 3, 4)
        x_stack = x_stack.reshape(B*nH*nW, C, 256, 256)
        return x_stack, (nH, nW)
    
    # added by xnf
    def inverse_img_stack(self, x: torch.Tensor, stack_shape, patch_size):
        nH, nW = stack_shape
        _, C, _, _ = x.shape
        B = int(x.shape[0] / (nH * nW))
        
        x = x.reshape(B, nH, nW, C, patch_size, patch_size)
        x_patch_vert = x.chunk(nH, dim=1)
        x_concat_vert = torch.concat(x_patch_vert, dim=4)
        x_patch_hori = x_concat_vert.chunk(nW, dim=2)
        x_concat_hori = torch.concat(x_patch_hori, dim=5)
        x_out = x_concat_hori.squeeze(1).squeeze(1).contiguous()
        return x_out

    # added by xnf
    @torch.no_grad()
    def forward_latent_concat(self, x: torch.Tensor):
        x_stack, stack_shape = self.make_img_stack(x)
        encoded_tokens = self.encode(x_stack)[1]["min_encoding_indices"]
        latent_stack_recon = self.decode_tokens_to_latent(encoded_tokens)
        latent_recon = self.inverse_img_stack(latent_stack_recon, stack_shape, 16)
        quantized_states_for_dec = torch.einsum(
            'nchw,cd->ndhw', latent_recon.softmax(1),
        self.pixel_quantize.embedding.weight)
        x_hat, y_hat = self.pixel_decoder.forward_with_latent(quantized_states_for_dec)
        return x_hat, y_hat