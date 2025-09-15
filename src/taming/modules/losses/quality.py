import torch
import torch.nn as nn
import lpips

from pytorch_msssim import MS_SSIM
from lpips import LPIPS
from DISTS_pytorch import DISTS

class Quality_Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.ms_ssim = MS_SSIM(data_range=1.0)
        self.lpips = LPIPS(net="alex").eval()
        self.dists = DISTS().eval()

    def psnr(self, a, b):
        mse = torch.mean((a - b) ** 2)
        return -10 * torch.log10(mse)

    @torch.no_grad()
    def forward(self, a, b):
        psnr = self.psnr(a, b).item()
        msssim = self.ms_ssim(a, b).item()
        lpips = self.lpips(a, b, normalize=True).item()
        dists = self.dists(a, b).item()
        return {
            "psnr": psnr,
            "msssim": msssim,
            "lpips": lpips,
            "dists": dists
        }