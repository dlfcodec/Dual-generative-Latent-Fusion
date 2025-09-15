# modified by Naifu Xue.

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from taming.util import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import ProductQuantizer


class Encoder_wo_attn(Encoder):
    def __init__(self, *, ch, out_ch, ch_mult=..., num_res_blocks, 
                 attn_resolutions, dropout=0, resamp_with_conv=True, 
                 in_channels, resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, 
                         attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=resamp_with_conv, 
                         in_channels=in_channels, resolution=resolution, z_channels=z_channels, double_z=double_z, **ignore_kwargs)
        self.mid.attn_1 = torch.nn.Identity()
        for i_level in range(self.num_resolutions):
            self.down.attn = torch.nn.ModuleList()


class Decoder_wo_attn(Decoder):
    def __init__(self, *, ch, out_ch, ch_mult=..., num_res_blocks, 
                 attn_resolutions, dropout=0, resamp_with_conv=True, 
                 in_channels, resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, 
                         attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=resamp_with_conv, 
                         in_channels=in_channels, resolution=resolution, z_channels=z_channels, give_pre_end=give_pre_end, **ignorekwargs)
        self.mid.attn_1 = torch.nn.Identity()
        for i_level in range(self.num_resolutions):
            self.up.attn = torch.nn.ModuleList()


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder_wo_attn(**ddconfig)
        self.decoder = Decoder_wo_attn(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        # for new lightning version
        self.automatic_optimization = False

        # calculate down sample factor
        self.downsampling_factor = 2 ** (self.encoder.num_resolutions - 1)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(f"[VQGAN wo Attn] Restored from {path}")
        print(f"[VQGAN wo Attn] missing_keys: {missing_keys}")
        print(f"[VQGAN wo Attn] unexpected_keys: {unexpected_keys}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)

        # get optimizers
        opt_ae, opt_disc = self.optimizers()

        # now optimize the codec
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx=0, 
                                        global_step=self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/ae_loss", aeloss, 
                 prog_bar=True, logger=True, on_step=True, on_epoch=False, batch_size=x.shape[0])
        self.log_dict(log_dict_ae, 
                      prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=x.shape[0])
        
        # manually optimize
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        # now optimize the discriminator
        if self.global_step > self.loss.discriminator_iter_start:
            xrec, qloss = self(x)
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx=1, 
                                                global_step=self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            discloss_log = log_dict_disc.pop("train/disc_loss")
            self.log("train/disc_loss", discloss_log, 
                     prog_bar=True, logger=True, on_step=True, on_epoch=False, batch_size=x.shape[0])
            self.log_dict(log_dict_disc, 
                          prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=x.shape[0])

            # manually optimize
            opt_disc.zero_grad()
            self.manual_backward(discloss)
            opt_disc.step()
        
        return None

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae.pop("val/rec_loss")

        self.log("saved_loss", rec_loss,    # this is for ckpt file name
                 prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        self.log("val/rec_loss", rec_loss, 
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        self.log("val/ae_loss", aeloss, 
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        self.log_dict(log_dict_ae, 
                      prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        self.log_dict(log_dict_disc, 
                      prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=x.shape[0])

        return rec_loss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModel_PQ(VQModel):
    def __init__(self, 
                 ddconfig, 
                 lossconfig, 
                 n_embed, 
                 n_seg,
                 embed_dim, 
                 ckpt_path=None, 
                 ignore_keys=[], 
                 image_key="image", 
                 colorize_nlabels=None, 
                 monitor=None, 
                 remap=None, 
                 sane_index_shape=False):
        super().__init__(
            ddconfig, lossconfig, n_embed, embed_dim, 
            None, ignore_keys, image_key, colorize_nlabels, 
            monitor, remap, sane_index_shape
        )
        self.quantize = ProductQuantizer(n_embed, n_seg, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)