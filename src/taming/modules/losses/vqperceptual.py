# modified by Naifu Xue.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from taming.modules.losses.lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from lpips import LPIPS as LPIPS_new


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", adaptive_disc_max=1.0e4):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.adaptive_disc_max = adaptive_disc_max

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, self.adaptive_disc_max).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0, device=rec_loss.device)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor, device=rec_loss.device),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


class VQLPIPSWithDiscriminator_sq_vq(VQLPIPSWithDiscriminator):
    def __init__(self, disc_start, codebook_weight=1, sq_weight=8.0, pixelloss_weight=1, 
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1, disc_weight=1, 
                 perceptual_weight=1, use_actnorm=False, disc_conditional=False, disc_ndf=64, 
                 disc_loss="hinge", adaptive_disc_max=10000):
        super().__init__(disc_start, codebook_weight, pixelloss_weight, 
                         disc_num_layers, disc_in_channels, disc_factor, disc_weight, perceptual_weight, 
                         use_actnorm, disc_conditional, disc_ndf, disc_loss, adaptive_disc_max)
        self.sq_weight = sq_weight
    
    def forward(self, codebook_loss, sq_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        loss, log = super().forward(codebook_loss, inputs, reconstructions, optimizer_idx,
                        global_step, last_layer, cond, split)
        if optimizer_idx == 0:
            sq_loss = sq_loss.mean()
            loss += self.sq_weight * sq_loss
            log["{}/sq_loss".format(split)] = sq_loss.detach()
            log["{}/sq_lambda".format(split)] = self.sq_weight
        return loss, log


class ConvNextPerceptualLoss(torch.nn.Module):
    def __init__(self, use_spatial=False):
        super().__init__()
        _IMAGENET_MEAN = [0.485, 0.456, 0.406]
        _IMAGENET_STD = [0.229, 0.224, 0.225]
        self.use_spatial = use_spatial

        self.convnext = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1).eval()
        self.register_buffer("imagenet_mean", torch.Tensor(_IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("imagenet_std", torch.Tensor(_IMAGENET_STD)[None, :, None, None])
        self.loss_fn = nn.MSELoss(reduction='mean')
        if use_spatial:
            del self.convnext.avgpool, self.convnext.classifier
            print("[VQGAN Loss] use spatial convnext loss")
        else:
            print("[VQGAN Loss] use logits convnext loss")

        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # Always in eval mode.
        self.eval()
        inputs = inputs * 0.5 + 0.5
        targets = targets * 0.5 + 0.5
        if self.use_spatial:
            pred_inputs = self.convnext.features((inputs - self.imagenet_mean) / self.imagenet_std)
            pred_targets = self.convnext.features((targets - self.imagenet_mean) / self.imagenet_std)
        else:
            inputs = torch.nn.functional.interpolate(inputs, size=224, mode="bilinear", align_corners=False, antialias=True)
            targets = torch.nn.functional.interpolate(targets, size=224, mode="bilinear", align_corners=False, antialias=True)
            pred_inputs = self.convnext((inputs - self.imagenet_mean) / self.imagenet_std)
            pred_targets = self.convnext((targets - self.imagenet_mean) / self.imagenet_std)
        loss = self.loss_fn(pred_inputs, pred_targets)
    
        return loss


class VQLPIPSWithDiscriminator_convnext(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0, convnext_weight=0.5,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", adaptive_disc_max=1.0e4, spatial_convnext=False):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.convnext_loss = ConvNextPerceptualLoss(use_spatial=spatial_convnext).eval()
        self.convnext_weight = convnext_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.adaptive_disc_max = adaptive_disc_max

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, self.adaptive_disc_max).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()

        if optimizer_idx == 0:
            # l1 loss
            rec_loss = torch.abs(inputs - reconstructions).mean()

            # lpips loss
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs, reconstructions).mean()
            else:
                p_loss = torch.zeros_like(inputs).mean()

            # convnext loss
            if self.convnext_weight > 0:
                conv_loss = self.convnext_loss(inputs, reconstructions).mean()
            else:
                conv_loss = torch.zeros_like(inputs).mean()

            # l1 + lpips loss, for adaptive gan weight
            nll_loss = rec_loss + self.perceptual_weight * p_loss

            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions)
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions, cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0, device=rec_loss.device)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + self.convnext_weight * conv_loss + d_weight * disc_factor * g_loss
            loss = loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor, device=rec_loss.device),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/convnext_loss".format(split): conv_loss.detach().mean(),
                   }
            return loss, log

        elif optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.detach())
                logits_fake = self.discriminator(reconstructions.detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

        else:
            raise NotImplementedError("Invalid optimizer_idx!")


class VQLPIPSWithDiscriminator_convnext_sq(VQLPIPSWithDiscriminator_convnext):
    def __init__(self, disc_start, codebook_weight=1, pixelloss_weight=1, convnext_weight=0.5, 
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1, disc_weight=1, 
                 perceptual_weight=1, use_actnorm=False, disc_conditional=False, disc_ndf=64, disc_loss="hinge", 
                 adaptive_disc_max=10000, sq_weight=1.0, spatial_convnext=False):
        super().__init__(disc_start, codebook_weight, pixelloss_weight, convnext_weight, 
                         disc_num_layers, disc_in_channels, disc_factor, disc_weight, perceptual_weight, 
                         use_actnorm, disc_conditional, disc_ndf, disc_loss, adaptive_disc_max, spatial_convnext)
        self.sq_weight = sq_weight

    def forward(self, codebook_loss, sq_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        loss, log = super().forward(codebook_loss, inputs, reconstructions, optimizer_idx,
                        global_step, last_layer, cond, split)
        if optimizer_idx == 0:
            sq_loss = sq_loss.mean()
            loss += self.sq_weight * sq_loss
            log["{}/sq_loss".format(split)] = sq_loss.detach()
            log["{}/sq_lambda".format(split)] = self.sq_weight
        return loss, log




class VQLPIPSWithDiscriminator_improveLPIPS(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", adaptive_disc_max=1.0e4, perceptual_alex_weight=0.1):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS_new(net='vgg').eval()
        self.perceptual_weight = perceptual_weight
        self.perceptual_loss_alex = LPIPS_new(net='alex').eval()
        self.perceptual_alex_weight = perceptual_alex_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"[INFO] VQLPIPSWithDiscriminator running with {disc_loss} loss. Alex ({perceptual_alex_weight:.2f}) + VGG ({perceptual_weight:.2f}) lpips.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.adaptive_disc_max = adaptive_disc_max

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, self.adaptive_disc_max).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous(), normalize=False)
            p_loss_alex = self.perceptual_loss_alex(inputs.contiguous(), reconstructions.contiguous(), normalize=False)
            rec_loss = rec_loss + self.perceptual_weight * p_loss + self.perceptual_alex_weight * p_loss_alex
        else:
            p_loss = torch.tensor([0.0], device=rec_loss.device)
            p_loss_alex = torch.tensor([0.0], device=rec_loss.device)

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0, device=rec_loss.device)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/p_loss_alex".format(split): p_loss_alex.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor, device=rec_loss.device),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


class VQLPIPSWithDiscriminator_improveLPIPS_sq_vq(VQLPIPSWithDiscriminator_improveLPIPS):
    def __init__(self, disc_start, codebook_weight=1, sq_weight=8.0, pixelloss_weight=1, 
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1, disc_weight=1, 
                 perceptual_weight=1, use_actnorm=False, disc_conditional=False, disc_ndf=64, 
                 disc_loss="hinge", adaptive_disc_max=10000, perceptual_alex_weight=0.1):
        super().__init__(disc_start, codebook_weight, pixelloss_weight, 
                         disc_num_layers, disc_in_channels, disc_factor, disc_weight, perceptual_weight, 
                         use_actnorm, disc_conditional, disc_ndf, disc_loss, adaptive_disc_max, perceptual_alex_weight)
        self.sq_weight = sq_weight
    
    def forward(self, codebook_loss, sq_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        loss, log = super().forward(codebook_loss, inputs, reconstructions, optimizer_idx,
                        global_step, last_layer, cond, split)
        if optimizer_idx == 0:
            sq_loss = sq_loss.mean()
            loss += self.sq_weight * sq_loss
            log["{}/sq_loss".format(split)] = sq_loss.detach()
            log["{}/sq_lambda".format(split)] = self.sq_weight
        return loss, log