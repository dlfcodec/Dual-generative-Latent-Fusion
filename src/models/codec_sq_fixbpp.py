import math
import random as rd
import numpy as np
import torchac
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from einops import rearrange
import torch.utils.checkpoint as cp

from taming.models.vqgan import VQModel as VQGAN
from taming.models.vqgan_wo_attn import VQModel as VQGAN_wo_Attn
from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator_sq_vq
from losses.feat_mse import FeatLoss_sq_vq
from blocks.swin_transformer import SwinBlock, Rearrange, rearrange
from blocks.conv_blocks import ConvNeXtBlock
from titok.quantizer import VectorQuantizer
from titok.titok import TiTok, TiTokEncoder, TiTokDecoder

from models.utils import _expand_token
from .cross_blocks import Interactive_crossAttn_type4
from .sq_bottleneck import Compressive_bottleneck_varbpp_type2 as Compressive_bottleneck

from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def print_rank0(information):
    print(information)


def get_swin(feat_width, num_layers, mlp_ratio=4.0, window_size=16, auto_bchw=True, inverse_shifted=False):
    assert feat_width % 64 == 0
    m_list = []
    if auto_bchw:
        m_list.append(Rearrange('b c h w -> b h w c'))
    for i in range(num_layers):
        shifted = not(bool(i % 2)) if inverse_shifted else bool(i % 2)
        relative_pos_embedding = False if inverse_shifted else bool(i==0)
        m_list.append(SwinBlock(feat_width, feat_width//64, 64, int(feat_width * mlp_ratio),
                                shifted=shifted, window_size=window_size, relative_pos_embedding=relative_pos_embedding))
    if auto_bchw:
        m_list.append(Rearrange('b h w c -> b c h w'))
    return nn.Sequential(*m_list)


class HybridEncoder(TiTokEncoder):
    def __init__(self, config, insert_pos, feat_width, num_attns=2, save_mem=False):
        super().__init__(config)
        # disable grad in TiTok's params
        self.requires_grad_(False)
        self.save_mem = save_mem
        if save_mem:
            print(f"[Hybrid Encoder] Enable checkpoint for memory saving!")

        # additional swin config
        self.feat_width = feat_width
        self.insert_pos = insert_pos
        self.cross_block_num = len(self.insert_pos)
        print(f"[Hybrid Encoder] current feat width: {feat_width}")
        print(f"[Hybrid Encoder] feat insert position: {insert_pos}")

        # pixel emb proj
        self.pix_emb_proj = nn.Conv2d(self.width, feat_width, kernel_size=1)
        
        inter_blocks = []
        feat_blocks = []
        for i in range(self.num_layers):
            if i in self.insert_pos:
                inter_blocks.append(Interactive_crossAttn_type4(self.width, feat_width, num_attns=num_attns, 
                                                                titok_patch_size=self.patch_size, 
                                                                feat_patch_size=self.patch_size, 
                                                                extra_titok_tokens=self.num_latent_tokens+1))
                feat_blocks.append(nn.Sequential(
                    get_swin(feat_width, num_layers=2),
                    ConvNeXtBlock(feat_width, feat_width, mlp_ratio=2.0, kernel_size=5),
                    ConvNeXtBlock(feat_width, feat_width, mlp_ratio=2.0, kernel_size=5)
                ))
            else:
                inter_blocks.append(nn.Identity())
                feat_blocks.append(nn.Identity())
        self.inter_blocks = nn.ModuleList(inter_blocks)
        self.feat_blocks = nn.ModuleList(feat_blocks)
        print(f"[Hybrid Encoder] num attn blocks per cross: {num_attns}")

        self.feat_in = get_swin(feat_width, num_layers=4)
        self.feat_out = nn.Sequential(
            get_swin(feat_width, num_layers=2),
            nn.Conv2d(feat_width, feat_width, kernel_size=2, stride=2),
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(feat_width),
            nn.Linear(feat_width, feat_width),
            Rearrange('b h w c -> b c h w'),
        )


    def set_to_zero(self):
        for m in self.inter_blocks:
            if hasattr(m, 'set_to_zero'):
                m.set_to_zero()


    def get_trainable_params(self, tune_titok=False):
        if tune_titok:
            p_list = list(self.parameters())
            print("[Hyper encoder] Tuning TiTok encoder's transformer!")
        else:
            p_list = list(self.pix_emb_proj.parameters()) +\
                    list(self.inter_blocks.parameters()) + list(self.feat_blocks.parameters()) +\
                    list(self.feat_in.parameters()) + list(self.feat_out.parameters())
        for p in p_list:
            p.requires_grad_(True)
        return p_list


    def forward(self, pixel_values, latent_tokens):
        # embedding
        x_emb = self.patch_embed(pixel_values)     # B C H W
        feat_emb = self.pix_emb_proj(x_emb)

        # make stack, to (B nH nW) C 16 16
        nH, nW = x_emb.shape[2] // 16, x_emb.shape[3] // 16
        stack_shape = (nH, nW)
        x = rearrange(x_emb, 'B C (nH h) (nW w) -> (B nH nW) C h w', h=16, w=16, nH=nH, nW=nW).contiguous()

        # reshape to (B nH nW) 256 width, e.g. NLD
        B = x.shape[0]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype) # shape = [*, grid ** 2 + 1, width]
        
        # concat latent tokens for quant in the sequence tail
        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)

        # feat init
        if self.save_mem:
            feat = cp.checkpoint(self.feat_in, feat_emb)
        else:
            feat = self.feat_in(feat_emb)

        # transform
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            if self.save_mem:
                # 1. transform in TiTok
                x = cp.checkpoint(self.transformer[i], x)

                # 2. extract information from titok
                if i in self.insert_pos:
                    feat, x = cp.checkpoint(self.inter_blocks[i], feat, x, stack_shape)
                    feat = cp.checkpoint(self.feat_blocks[i], feat)
            else:
                # 1. transform in TiTok
                x = self.transformer[i](x)
            
                # 2. extract information from titok
                if i in self.insert_pos:
                    feat, x = self.inter_blocks[i](feat, x, stack_shape)
                    feat = self.feat_blocks[i](feat)

        x = x.permute(1, 0, 2)  # LND -> NLD

        # output of titok
        latent_tokens = x[:, 1+self.grid_size**2:]
        latent_tokens = self.ln_post(latent_tokens)

        # fake 2D shape of titok
        latent_tokens = latent_tokens.reshape(B, self.width, self.num_latent_tokens, 1)
        latent_tokens = self.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(B, self.token_size, 1, self.num_latent_tokens)

        if self.save_mem:
            feat = cp.checkpoint(self.feat_out, feat)
        else:
            feat = self.feat_out(feat)
        return latent_tokens, feat, stack_shape


class HybridDecoder(TiTokDecoder):
    def __init__(self, config, insert_pos, feat_width, num_attns=2, save_mem=False):
        super().__init__(config)
        # disable grad in TiTok's params
        self.requires_grad_(False)
        self.save_mem = save_mem
        if save_mem:
            print(f"[Hybrid Decoder] Enable checkpoint for memory saving!")

        self.ffn = nn.Identity()    # remove original ffn

        # additional swin config
        self.insert_pos = insert_pos
        self.cross_block_num = len(self.insert_pos)
        print(f"[Hybrid Decoder] current feat width: {feat_width}")
        print(f"[Hybrid Decoder] feat insert position: {insert_pos}")

        self.init_feat_up = nn.Sequential( 
            nn.Conv2d(feat_width, feat_width*4, kernel_size=1),
            nn.PixelShuffle(2),
            get_swin(feat_width, num_layers=4)
        )

        inter_blocks = []
        feat_blocks = []
        for i in range(self.num_layers):
            if i in self.insert_pos:
                inter_blocks.append(Interactive_crossAttn_type4(self.width, feat_width, num_attns=num_attns, 
                                                                titok_patch_size=self.patch_size, 
                                                                feat_patch_size=self.patch_size,
                                                                extra_titok_tokens=self.num_latent_tokens+1))
                feat_blocks.append(nn.Sequential(
                    get_swin(feat_width, num_layers=2),
                    ConvNeXtBlock(feat_width, feat_width, mlp_ratio=2.0, kernel_size=5),
                    ConvNeXtBlock(feat_width, feat_width, mlp_ratio=2.0, kernel_size=5)
                ))
            else:
                inter_blocks.append(nn.Identity())
                feat_blocks.append(nn.Identity())
        self.inter_blocks = nn.ModuleList(inter_blocks)
        self.feat_blocks = nn.ModuleList(feat_blocks)
        print(f"[Hybrid Decoder] num attn blocks per cross: {num_attns}")


    def set_to_zero(self):
        for m in self.inter_blocks:
            if hasattr(m, 'set_to_zero'):
                m.set_to_zero()


    def get_trainable_params(self, tune_titok=False):
        if tune_titok:
            p_list = list(self.parameters())
            print("[Hyper decoder] Tuning TiTok decoder's transformer!")
        else:
            p_list = list(self.init_feat_up.parameters()) + list(self.inter_blocks.parameters()) +\
                    list(self.feat_blocks.parameters())
        for p in p_list:
            p.requires_grad_(True)
        return p_list
    

    def forward(self, z_quantized, h_quantized, stack_shape, **kw_args):
        # get shape
        nH, nW = stack_shape
        N, C, H, W = z_quantized.shape
        assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD

        # decoder emb in titok
        x = self.decoder_embed(x)
        batchsize, seq_len, _ = x.shape

        # concat masked tokens
        mask_tokens = self.mask_token.repeat(batchsize, self.grid_size**2, 1).to(x.dtype)
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)

        # for res feat branch
        if self.save_mem:
            feat = cp.checkpoint(self.init_feat_up, h_quantized)
        else:
            feat = self.init_feat_up(h_quantized)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            if self.save_mem:
                # 1. titok transformer
                x = cp.checkpoint(self.transformer[i], x)

                # 2. extract information from titok
                if i in self.insert_pos:
                    feat, x = cp.checkpoint(self.inter_blocks[i], feat, x, stack_shape)
                    feat = cp.checkpoint(self.feat_blocks[i], feat)
            else:
                # 1. titok transformer
                x = self.transformer[i](x)
            
                # 2. extract information from titok
                if i in self.insert_pos:
                    feat, x = self.inter_blocks[i](feat, x, stack_shape)
                    feat = self.feat_blocks[i](feat)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+self.grid_size**2] # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        x = rearrange(x, '(B nH nW) C h w -> B C (nH h) (nW w)', 
                        nH=nH, nW=nW, h=self.patch_size, w=self.patch_size).contiguous()
        return x, feat
    

class Hybrid_Codec(nn.Module):
    def __init__(self, config, in_pos_enc, in_pos_dec, feat_width, quant_dim, n_enc_attn, save_mem=False):
        super().__init__()
        self.config = config
        self.encoder = HybridEncoder(config, in_pos_enc, feat_width, num_attns=n_enc_attn, save_mem=save_mem)
        self.decoder = HybridDecoder(config, in_pos_dec, feat_width, num_attns=n_enc_attn, save_mem=save_mem)

        # TiTok's default part
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        self.latent_tokens.requires_grad_(False)
        
        self.apply(self._init_weights)

        self.quantize = VectorQuantizer(
            codebook_size=config.model.vq_model.codebook_size,
            token_size=config.model.vq_model.token_size,
            commitment_cost=config.model.vq_model.commitment_cost,
            use_l2_norm=config.model.vq_model.use_l2_norm,)
        self.quantize.requires_grad_(False)

        # additional part
        self.quantize_feat = Compressive_bottleneck(feat_width, quant_dim, bpp_num=1)
        self.quantize_feat.requires_grad_(True)

        # set zero conv
        self.encoder.set_to_zero()
        self.decoder.set_to_zero()


    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.05)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.05)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def init_from_titok_sd(self, sd):
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        # print(f"[Hybrid Codec] init params from statedict...")
        # print(f"missing_keys: {missing_keys}")
        # print(f"unexpected_keys: {unexpected_keys}")


    def get_trainable_params(self, tune_titok=False):
        if tune_titok:
            p_list = list(self.parameters())
            print("[Codec] tunning titok's parameters!")
        else:
            p_list = self.encoder.get_trainable_params(tune_titok=False) +\
                     self.decoder.get_trainable_params(tune_titok=False) +\
                    list(self.quantize_feat.parameters())
        for p in p_list:
            p.requires_grad_(True)
        return p_list


    def encode(self, x):
        z, h, stack_shape = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        z_quantized, z_result_dict = self.quantize(z)
        h_quantized, h_result_dict = self.quantize_feat.forward(h, (x.shape[2], x.shape[3]), q_idx=0)
        return {
            'z_quantized': z_quantized, 'z_result_dict': z_result_dict,
            'h_quantized': h_quantized, 'h_result_dict': h_result_dict, 
            'stack_shape': stack_shape
        }


    def decode(self, z_quantized, h_quantized, stack_shape, **kw_args):
        titok_hat, feat_hat = self.decoder(z_quantized, h_quantized, stack_shape)
        return titok_hat, feat_hat


    def forward(self, x):
        out = self.encode(x)
        titok_hat, feat_hat = self.decode(**out)
        out['titok_hat'] = titok_hat
        out['feat_hat'] = feat_hat
        return out


class FeatMerge(nn.Module):
    def __init__(self, titok_width, feat_width, n_embed=256, save_mem=False):
        super().__init__()
        self.save_mem = save_mem
        if save_mem:
            print(f"[FeatMerge] Enable checkpoint for memory saving!")

        self.feat_in = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            get_swin(feat_width, num_layers=2, auto_bchw=False)
        )
        self.titok_in = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            get_swin(titok_width, num_layers=2, auto_bchw=False)
        )
        inner_width = 1024
        self.merge = nn.Sequential(
            nn.Linear(titok_width + feat_width, titok_width*2),
            nn.LayerNorm(titok_width*2),
            nn.SiLU(),
            nn.Linear(titok_width*2, inner_width),
            get_swin(inner_width, num_layers=4, auto_bchw=False)
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(inner_width),
            nn.Linear(inner_width, 2*inner_width),
            nn.Tanh(),
            nn.Linear(2*inner_width, n_embed),
            Rearrange('b h w c -> b c h w')
        )


    def forward(self, titok, feat):
        # proj
        if self.save_mem:
            titok = cp.checkpoint(self.titok_in, titok)
            feat = cp.checkpoint(self.feat_in, feat)
            h = cp.checkpoint(self.merge, torch.cat([titok, feat], dim=-1))
            logits = cp.checkpoint(self.ffn, h)
        else:
            titok = self.titok_in(titok)
            feat = self.feat_in(feat)
            h = self.merge(torch.cat([titok, feat], dim=-1))
            logits = self.ffn(h)
        return logits
    

class Codec(pl.LightningModule):
    def __init__(self, embed_dim, feat_dim, in_pos_enc, in_pos_dec, n_attn,
                 config, vqganconfig, imglossconfig, featlossconfig, training_strategy,
                 monitor="", ckpt_path=None, ignore_keys=[], titok_pretrain_path=None, tune_titok=False, 
                 no_attn_vqgan=False, save_mem=False):
        super().__init__()
        # lightning configs
        self.image_key = "image"
        self.automatic_optimization = False
        if monitor is not None:
            self.monitor = monitor

        # variable bpp lambda config
        self.init_training_strategy(training_strategy)

        # load titok and hybrid model
        self.config = config
        titok = TiTok(config)
        if titok_pretrain_path:
            titok.load_state_dict(torch.load(titok_pretrain_path, map_location="cpu"), strict=True)
        else:
            print("[Warning] titok_pretrain_path is None!")
        
        self.hybrid_codec = Hybrid_Codec(config, in_pos_enc, in_pos_dec, feat_dim, embed_dim, n_attn, save_mem=save_mem)
        self.hybrid_codec.init_from_titok_sd(titok.state_dict())
        del titok
        self.tune_titok = tune_titok
        
        # vqgan
        self.vqgan = VQGAN_wo_Attn(**vqganconfig) if no_attn_vqgan else VQGAN(**vqganconfig)
        self.vqgan.quantize.sane_index_shape = True
        self.vqgan.encoder.requires_grad_(False)
        self.vqgan.quant_conv.requires_grad_(False)

        # prior fusion
        self.prior_fusion = FeatMerge(self.hybrid_codec.decoder.width, feat_dim, self.vqgan.quantize.n_e)

        # loss define
        self.feat_loss = FeatLoss_sq_vq(**featlossconfig)
        self.img_loss = VQLPIPSWithDiscriminator_sq_vq(**imglossconfig)
        self.img_loss.load_state_dict(self.vqgan.loss.state_dict(), strict=True)
        del self.vqgan.loss
        self.learning_rate = training_strategy['learning_rate']

        # load ckpt
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.is_set_torchac = False

        print("[INFO] ControlTiTok codec v1-7 SQ fixed bpp, f16 VQGAN, FINAL model")


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        print(f"missing_keys: {missing_keys}")
        print(f"unexpected_keys: {unexpected_keys}")


    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(self.hybrid_codec.get_trainable_params(self.tune_titok)+
                                list(self.prior_fusion.parameters())+
                                list(self.vqgan.quantize.parameters())+
                                list(self.vqgan.post_quant_conv.parameters())+
                                list(self.vqgan.decoder.parameters()),
                                lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.img_loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc]


    def init_training_strategy(self, training_strategy):
        """Set training strategy for each stage and its lambda setting
        """
        strategy_names = ['feat_wo_bpp', 'feat', 'pix']
        self.strategies = []
        for i in range(3):
            self.strategies += [
                {
                    'strategy': strategy_names[i],
                    'init_lmbda_idx': training_strategy[f'stage{i}']['init_lmbda_idx'],
                    'lmbda_list': training_strategy[f'stage{i}']['lmbda_list'],
                    'bpp_upper': training_strategy[f'stage{i}']['bpp_upper'],
                    'bpp_lower': training_strategy[f'stage{i}']['bpp_lower'],
                } for j in range(training_strategy[f'stage{i}']['epoch_num'])
            ]
        # record current epoch for continue training
        self.last_strategy = 'feat_wo_bpp'
        self.current_strategy = 'feat_wo_bpp'
        start_epoch = training_strategy['start_epoch']
        self.epoch_for_strategy = nn.Parameter(torch.tensor(start_epoch, dtype=torch.int32), requires_grad=False)
        self.lmbda_idx = nn.Parameter(torch.tensor(self.strategies[start_epoch]['init_lmbda_idx'], dtype=torch.int32, device=self.device),
                                    requires_grad=False)
        self.lmbda_list = nn.Parameter(torch.tensor(self.strategies[start_epoch]['lmbda_list'], dtype=torch.float32, device=self.device),
                                    requires_grad=False)


    def set_lmbda_to_this(self):
        self.img_loss.sq_weight = self.lmbda_list[self.lmbda_idx.item()].item()
        self.feat_loss.sq_weight = self.lmbda_list[self.lmbda_idx.item()].item()
    

    def on_train_epoch_start(self) -> None:
        """update training strategy before starting a new epoch
        """
        # 1. change training strategy 
        cur_epoch = self.epoch_for_strategy.item()
        self.current_strategy = self.strategies[cur_epoch]['strategy']
        if self.current_strategy in ['feat_wo_bpp', 'feat']:
            self.vqgan.quantize.requires_grad_(False)
            self.vqgan.post_quant_conv.requires_grad_(False)
            self.vqgan.decoder.requires_grad_(False)
            print_rank0(f"[Training] epoch {cur_epoch}, current strategy: {self.current_strategy}, fix vqgan's parameter")
        elif self.current_strategy in ['pix']:
            self.vqgan.quantize.requires_grad_(True)
            self.vqgan.post_quant_conv.requires_grad_(True)
            self.vqgan.decoder.requires_grad_(True)
            print_rank0(f"[Training] epoch {cur_epoch}, current strategy: {self.current_strategy}, tunning vqgan's parameter")

        # 2. set lambda
        if self.current_strategy != self.last_strategy:
            self.lmbda_idx.set_(torch.tensor(self.strategies[cur_epoch]['init_lmbda_idx'], dtype=torch.int32, device=self.device))
            self.lmbda_list.set_(torch.tensor(self.strategies[cur_epoch]['lmbda_list'], dtype=torch.float32, device=self.device))
            print_rank0(f"[Training] epoch {cur_epoch}, reset start lambda idx: {self.lmbda_idx.item()}, reset lambda list: {self.lmbda_list.cpu().tolist()}")

        # 3. checkpointing if change stage
        if self.current_strategy != self.last_strategy and self.current_epoch > 0:
            save_path = self.trainer.checkpoint_callback.dirpath
            save_path = f"{save_path}/{self.last_strategy}_epo_for_strategy_{cur_epoch-1}.ckpt"
            self.trainer.save_checkpoint(save_path)
            print_rank0(f"[Training] finish strategy {self.last_strategy}, save checkpoint")

        return super().on_train_epoch_start()


    def on_train_epoch_end(self) -> None:
        """update epoch counter and record last strategy
        """
        self.epoch_for_strategy += 1
        self.last_strategy = self.current_strategy
        print_rank0(f"[Training] epoch {self.epoch_for_strategy.item()} finish!")
        return super().on_train_epoch_end()


    def on_validation_epoch_start(self) -> None:
        """reset average metric before validation
        """
        cur_epoch = self.epoch_for_strategy.item()
        print_rank0(f"[Training] epoch {cur_epoch}, start validation!")

        # reset containers
        self.current_val_bpp = torch.zeros((1), device=self.device, requires_grad=False)
        self.val_batch_num = 0
        return super().on_validation_epoch_start()


    def on_validation_epoch_end(self) -> None:
        """calculate average validation metric
        """
        # skip sanity checking
        if self.trainer.sanity_checking:
            return super().on_validation_epoch_end()
        
        # skip wo bpp
        if self.current_strategy == "feat_wo_bpp":
            return super().on_validation_epoch_end()

        cur_epoch = self.epoch_for_strategy.item()

        # calculate average bpp with each lambda
        self.current_val_bpp /= self.val_batch_num
        print_rank0(f"[Training] epoch {cur_epoch}, validation bpp: {self.current_val_bpp.item()}")

        # get param
        bpp_lower = self.strategies[cur_epoch]['bpp_lower']
        bpp_upper = self.strategies[cur_epoch]['bpp_upper']

        # bpp adaption scheme
        if self.current_val_bpp > bpp_upper:
            print_rank0(f"[Adjusting lambda] val bpp > tar high bpp")
            self.lmbda_idx.set_(torch.clamp(self.lmbda_idx + 1, min=0, max=self.lmbda_list.shape[0]-1))
            print_rank0(f"[Adjusting lambda] set new lambda idx to {self.lmbda_idx.item()}")
        elif self.current_val_bpp < bpp_lower:
            print_rank0(f"[Adjusting lambda] val bpp < tar low bpp")
            self.lmbda_idx.set_(torch.clamp(self.lmbda_idx - 1, min=0, max=self.lmbda_list.shape[0]-1))
            print_rank0(f"[Adjusting lambda] set new lambda idx to {self.lmbda_idx.item()}")
        
        return super().on_validation_epoch_end()


    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        _, bpp_this = outputs
        bpp_reduced = self.trainer.strategy.reduce(bpp_this, reduce_op='mean')
        self.val_batch_num += 1
        self.current_val_bpp += bpp_reduced
        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)


    @torch.no_grad()
    def encode_to_vqgan(self, x):
        h = self.vqgan.encoder(x)
        h = self.vqgan.quant_conv(h)
        h_quantized, _, info = self.vqgan.quantize(h)
        return h_quantized, info[2]


    def decode_to_latent(self, titok_hat, feat_hat, **kw_args):
        logits = self.prior_fusion(titok_hat, feat_hat)
        quantized_latent = torch.einsum(
            'nchw,cd->ndhw', logits.softmax(1),
            self.vqgan.quantize.embedding.weight)
        return quantized_latent, logits
    
    
    def decode_to_image(self, quantized_latent):
        quantized_latent = self.vqgan.post_quant_conv(quantized_latent)
        x_hat = self.vqgan.decoder(quantized_latent)
        return x_hat


    def forward(self, x, need_full_decode=True):
        enc_info = self.hybrid_codec(x * 0.5 + 0.5)   # send x with [0, 1] to titok
        vqgan_latent, logits = self.decode_to_latent(titok_hat=enc_info['titok_hat'], 
                                                     feat_hat=enc_info['feat_hat'])
        
        if need_full_decode:
            x_hat = self.decode_to_image(vqgan_latent)
        else:
            x_hat = None

        # quant loss
        vq_loss = enc_info['z_result_dict']['quantizer_loss']
        bpp_loss = enc_info['h_result_dict']['bpp']
        bpp_hard_quant = enc_info['h_result_dict']['bpp_direct']

        return {"x": x, "x_hat": x_hat, "bpp_loss": bpp_loss, "vq_loss": vq_loss,
                "logits": logits, "vqgan_latent": vqgan_latent, "bpp_hard_quant": bpp_hard_quant}


    def get_input(self, batch, k):
        x = batch[k]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x


    def get_last_layer(self):
        return self.vqgan.decoder.conv_out.weight
    

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)

        # get optimizers
        opt_ae, opt_disc = self.optimizers()

        # set lambda
        self.set_lmbda_to_this()

        # first training stage: align the latent
        if self.current_strategy in ['feat_wo_bpp', 'feat']:
            latent_label, indices_label = self.encode_to_vqgan(x)
            out = self(x, need_full_decode=False)
            align_loss, log_dict_align = self.feat_loss.forward(
                feat_in=out['vqgan_latent'], logits_in=out['logits'], feat_target=latent_label, label_target=indices_label,
                vq_loss=out['vq_loss'], sq_loss=out['bpp_loss']
            )

            # manually optimize
            opt_ae.zero_grad()
            self.manual_backward(align_loss)
            opt_ae.step()

            self.log(f"train/align_loss", align_loss, 
                     prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=x.shape[0])
            self.log(f"train/lambda", self.feat_loss.sq_weight, 
                     prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=x.shape[0])
            self.log(f"train/bpp", out['bpp_loss'], 
                     prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=x.shape[0])
            self.log(f"train/bpp_hard_quant", out['bpp_hard_quant'], 
                     prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=x.shape[0])
            self.log(f"train/weighted_bpp", out['bpp_loss'] * self.feat_loss.sq_weight, 
                     prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=x.shape[0])
            self.log_dict(log_dict_align, 
                          prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=x.shape[0])

            return None

        elif self.current_strategy in ['pix']:
            # now optimize the codec
            self.img_loss.discriminator.requires_grad_(False)
            out = self(x, need_full_decode=True)
            aeloss, log_dict_ae = self.img_loss.forward(out['vq_loss'], out['bpp_loss'], out['x'], out['x_hat'], optimizer_idx=0, 
                                            global_step=self.global_step, last_layer=self.get_last_layer(), split=f"train")
            self.log(f"train/ae_loss", aeloss, 
                    prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=x.shape[0])
            self.log(f"train/lambda", self.img_loss.sq_weight, 
                    prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=x.shape[0])
            self.log(f"train/bpp", out['bpp_loss'], 
                     prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=x.shape[0])
            self.log(f"train/bpp_hard_quant", out['bpp_hard_quant'], 
                     prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=x.shape[0])
            self.log(f"train/weighted_bpp", out['bpp_loss'] * self.img_loss.sq_weight, 
                     prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=x.shape[0])
            self.log_dict(log_dict_ae, 
                        prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=x.shape[0])

            # manually optimize
            opt_ae.zero_grad()
            self.manual_backward(aeloss)
            opt_ae.step()

            # now optimize the discriminator
            if self.global_step > self.img_loss.discriminator_iter_start:
                self.img_loss.discriminator.requires_grad_(True)
                discloss, log_dict_disc = self.img_loss(out['vq_loss'], out['bpp_loss'], out['x'], out['x_hat'], optimizer_idx=1, 
                                            global_step=self.global_step, last_layer=self.get_last_layer(), split=f"train")
                discloss_log = log_dict_disc.pop(f"train/disc_loss")
                self.log(f"train/disc_loss", discloss_log, 
                        prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=x.shape[0])
                self.log_dict(log_dict_disc, 
                            prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=x.shape[0])

                # manually optimize
                opt_disc.zero_grad()
                self.manual_backward(discloss)
                opt_disc.step()

        else:
            raise NotImplementedError()

        return None


    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)

        # ser lmbda
        self.set_lmbda_to_this()
        
        # encode
        latent_label, indices_label = self.encode_to_vqgan(x)
        out = self(x, need_full_decode=True)

        # loss
        align_loss, log_dict_align = self.feat_loss.forward(
            feat_in=out['vqgan_latent'], logits_in=out['logits'], feat_target=latent_label, label_target=indices_label,
            vq_loss=out['vq_loss'], sq_loss=out['bpp_loss']
        )
        aeloss, log_dict_ae = self.img_loss(out['vq_loss'], out['bpp_loss'], out['x'], out['x_hat'], optimizer_idx=0, 
                                        global_step=self.global_step, last_layer=self.get_last_layer(), split=f"val")
        discloss, log_dict_disc = self.img_loss(out['vq_loss'], out['bpp_loss'], out['x'], out['x_hat'], optimizer_idx=1, 
                                        global_step=self.global_step, last_layer=self.get_last_layer(), split=f"val")
        rec_loss = log_dict_ae.pop(f"val/rec_loss")

        self.log(f"val/rec_loss", rec_loss, 
                prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        self.log(f"val/ae_loss", aeloss, 
                prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        self.log(f"val/align_loss", align_loss, 
                prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        self.log(f"val/bpp", out['bpp_loss'], 
                prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        self.log_dict(log_dict_ae, 
                    prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        self.log_dict(log_dict_disc, 
                    prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        self.log_dict(log_dict_align, 
                    prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=x.shape[0])

        # ensure the loss for save is small in stage 3.
        loss_for_save = rec_loss + self.img_loss.sq_weight * out['bpp_loss'] * 2.0      # 2.0 is fix factor, for hard quant bpp.
        if self.current_strategy not in ['pix']:
            loss_for_save += 100.0

        # this is for ckpt file name
        self.log("saved_loss", loss_for_save,    
                prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        return loss_for_save, out['bpp_loss']


    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        out = self(x, need_full_decode=True)
        log_img_dict = {'x': out['x'], 'x_hat': out['x_hat']}
        return log_img_dict


    def set_torchac(self):
        self.is_set_torchac = True
        titok_codebook_size = self.hybrid_codec.config.model.vq_model.codebook_size
        self.titok_prob_table = torch.ones((titok_codebook_size), device=self.device, dtype=torch.float32) / titok_codebook_size
        self.titok_prob_cdf = torch.zeros((titok_codebook_size+1), device="cpu")
        self.titok_prob_cdf[1:] = torch.cumsum(self.titok_prob_table, dim=0)


    @torch.no_grad()
    def encode_only(self, x):
        if not self.is_set_torchac:
            self.set_torchac()
        
        img_shape = (x.shape[2], x.shape[3])
        x_01 = x * 0.5 + 0.5
        z, h, stack_shape = self.hybrid_codec.encoder(pixel_values=x_01, latent_tokens=self.hybrid_codec.latent_tokens)

        # compress z
        z_quantized, z_result_dict = self.hybrid_codec.quantize(z)
        z_indices_shape = z_quantized.shape
        z_indices = z_result_dict['min_encoding_indices'].flatten().cpu()
        token_length = z_indices.shape[0]
        z_cdf_shaped = self.titok_prob_cdf.unsqueeze(0).repeat(token_length, 1).cpu()
        z_bit_stream = torchac.encode_float_cdf(z_cdf_shaped, z_indices.to(dtype=torch.int16))

        # compress h
        feat_shape = h.shape
        h_bit_stream = self.hybrid_codec.quantize_feat.compress(h, q_idx=0)
        
        return {
            "z_bit_stream": z_bit_stream, 
            "h_bit_stream": h_bit_stream,
            "img_shape": img_shape,
            "feat_shape": feat_shape,
            "stack_shape": stack_shape,
            "token_length": token_length,
            "z_indices_shape": z_indices_shape
        }


    @torch.no_grad()
    def decode_only(self, z_bit_stream, h_bit_stream, img_shape, feat_shape, stack_shape, token_length, z_indices_shape):
        # decode z
        z_cdf_shaped = self.titok_prob_cdf.unsqueeze(0).repeat(token_length, 1).cpu()
        z_hat = torchac.decode_float_cdf(z_cdf_shaped, z_bit_stream)
        z_hat = z_hat.to(device=self.device, dtype=torch.int32)
        z_hat = self.hybrid_codec.quantize.get_codebook_entry(z_hat)
        z_hat = rearrange(z_hat, "(l n) c -> l c n", l=z_indices_shape[0], c=z_indices_shape[1], n=z_indices_shape[3])
        z_hat = z_hat.unsqueeze(2).contiguous()
        z_hat = torch.nn.functional.normalize(z_hat, dim=1)

        # decode h
        h_hat = self.hybrid_codec.quantize_feat.decompress(h_bit_stream, feat_shape, q_idx=0)

        # reconstruction image
        titok_hat, feat_hat = self.hybrid_codec.decoder(z_hat, h_hat, stack_shape)
        vqgan_latent, logits = self.decode_to_latent(titok_hat=titok_hat, feat_hat=feat_hat)
        x_hat = self.decode_to_image(vqgan_latent)
        return x_hat.clamp(-1.0, 1.0)


    @torch.no_grad()
    def encode_decode(self, x, original_shape):
        """ encode and decode an image, simulate write and read from a bitstream
        """
        enc_result = self.encode_only(x)
        x_hat = self.decode_only(**enc_result)
        z_bits = len(enc_result['z_bit_stream']) * 8
        h_bits = len(enc_result['h_bit_stream']) * 8
        overhead_bits = 8 * 6       # 6 Bytes, 4 Bytes for height and width, 2 Bytes for token stream length.
                                    # since feat_shape, stack_shape and z_indices_shape can be inferred from img_shape.

        h, w = original_shape
        bpp_dict = {
            'z_bpp': z_bits / (h * w),
            'h_bpp': h_bits / (h * w),
            'overhead_bpp': overhead_bits / (h * w),
            'total_bpp': (z_bits + h_bits + overhead_bits) / (h * w)
        }

        return x_hat, bpp_dict, enc_result