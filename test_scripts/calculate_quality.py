import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
from PIL import Image

import lpips
from DISTS_pytorch import DISTS
from pytorch_msssim import ms_ssim
from torchmetrics.image.fid import FrechetInceptionDistance

import argparse
import os, sys
sys.path.append(os.getcwd())

# enable print all dataframe
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def image_to_255_scale(image, dtype = None):
    """
    Helper function for converting a floating point image to 255 scale.

    The input image is expected to be in the range [0.0, 1.0]. If it is outside
    this range, the function throws an error.

    Args:
        image: A 4-D PyTorch tensor.
        dtype: Output datatype. If not passed, the image is of the same dtype
            as the input.

    Returns:
        The image in [0, 255] scale.
    """
    if image.max() > 1.0:
        raise ValueError("Unexpected image max > 1.0")
    if image.min() < 0.0:
        raise ValueError("Unexpected image min < 0.0")

    image = torch.round(image * 255.0)

    if dtype is not None:
        image = image.to(dtype)

    return image

def update_patch_fid(
    input_images, pred,
    fid_metric = None,
    fid_swav_metric = None,
    kid_metric = None,
    patch_size = 256,
    split_patch_num = 2,
):
    """
    Update FID and KID metrics with patch-based calculation.

    This implements the FID/256 (and KID/256) method described in the following
    paper:

    High-Fidelity Generative Image Compression
    Fabian Mentzer, George D. Toderici, Michael Tschannen, Eirikur Agustsson

    First, a user defines a torchmetric class for FID or KID. Then, this
    function can be used to update the metrics using the FID/256 calculation.

    This method gives more stable FID calculations for small counts of
    high-resolution images, such as those in the CLIC2020 dataset. Each image
    is divided up into a grid of non-overlapping patches, and the normal FID
    calculation is run treating each patch as an image. Then, the calculation
    is re-run a second time with a patch/2 shift.

    Args:
        input_images: The ground truth images in [0.0, 1.0] range.
        pred: The compressed images in [0.0, 1.0] range.
        fid_metric: A torchmetric for calculationg FID.
        fid_swav_metric: A torchmetric for calculating FID with the SwAV
            backbone.
        kid_metric: A torchmetric for calculating KID.
        patch_size: The patch size to use for dividing up each image.

    Returns:
        The number of patches (metric are updated in-place). The number of
        patches can be used as debugging signal.
    """
    if fid_metric is None and kid_metric is None and fid_swav_metric is None:
        raise ValueError("At least one metric must not be None.")

    # this applies the FID/KID calculations from Mentzer 2020
    real = image_to_255_scale(
        F.unfold(input_images, kernel_size=patch_size, stride=patch_size)
        .permute(0, 2, 1)
        .reshape(-1, 3, patch_size, patch_size),
        dtype=torch.uint8,
    )
    fake = image_to_255_scale(
        F.unfold(pred, kernel_size=patch_size, stride=patch_size)
        .permute(0, 2, 1)
        .reshape(-1, 3, patch_size, patch_size),
        dtype=torch.uint8,
    )
    patch_count = real.shape[0]
    if fid_metric is not None:
        fid_metric.update(real, real=True)
        fid_metric.update(fake, real=False)
    if fid_swav_metric is not None:
        fid_swav_metric.update(real, real=True)
        fid_swav_metric.update(fake, real=False)
    if kid_metric is not None:
        kid_metric.update(real, real=True)
        kid_metric.update(fake, real=False)

    num_y, num_x = input_images.shape[2], input_images.shape[3]

    unit = patch_size // split_patch_num
    for unit_i in range(1, split_patch_num):
        limit_size = (2. - unit_i / split_patch_num) * patch_size
        if num_y >= limit_size and num_x >= limit_size:
            real = image_to_255_scale(
                F.unfold(
                    input_images[:, :, unit * unit_i:, unit * unit_i:],
                    kernel_size=patch_size,
                    stride=patch_size,
                )
                .permute(0, 2, 1)
                .reshape(-1, 3, patch_size, patch_size),
                dtype=torch.uint8,
            )
            fake = image_to_255_scale(
                F.unfold(
                    pred[:, :, unit * unit_i:, unit * unit_i:],
                    kernel_size=patch_size,
                    stride=patch_size,
                )
                .permute(0, 2, 1)
                .reshape(-1, 3, patch_size, patch_size),
                dtype=torch.uint8,
            )
            patch_count += real.shape[0]
            if fid_metric is not None:
                fid_metric.update(real, real=True)
                fid_metric.update(fake, real=False)
            if fid_swav_metric is not None:
                fid_swav_metric.update(real, real=True)
                fid_swav_metric.update(fake, real=False)
            if kid_metric is not None:
                kid_metric.update(real, real=True)
                kid_metric.update(fake, real=False)

    return patch_count


class OnlyImageFolder_compare(Dataset):
    def __init__(self, ref_path, recon_path, recon_pfx='.png'):
        self.ref_path = ref_path
        self.recon_path = recon_path
        self.recon_pfx = recon_pfx
        self.ref_images = sorted(list(os.listdir(ref_path)))
        self.dataset_length = len(self.ref_images)
        print(f"Datasets: {self.dataset_length} images")

    def __getitem__(self, index):
        img_name = self.ref_images[index]
        img_name_recon = os.path.splitext(img_name)[0] + self.recon_pfx

        ref_img_path = os.path.join(self.ref_path, img_name)
        ref_img = Image.open(ref_img_path).convert("RGB")
        ref_img = np.array(ref_img).transpose(2, 0, 1)
        ref_img = torch.as_tensor(ref_img.astype(np.float32) / 255.0, dtype=torch.float32)

        recon_img_path = os.path.join(self.recon_path, img_name_recon)
        recon_img = Image.open(recon_img_path).convert("RGB")
        recon_img = np.array(recon_img).transpose(2, 0, 1)
        recon_img = torch.as_tensor(recon_img.astype(np.float32) / 255.0, dtype=torch.float32)

        return ref_img, recon_img, img_name

    def __len__(self):
        return self.dataset_length


class test_two_folder():
    def __init__(self, recon_path, ref_path, patch_size, split_patch_num, exp_name=None):
        # data
        self.exp_name = exp_name
        self.recon_path = recon_path
        self.ref_path = ref_path
        self.imgset = OnlyImageFolder_compare(ref_path, recon_path)
        self.imgloader = DataLoader(self.imgset, batch_size=1, shuffle=False)

        # eval models
        self.lpips_metric = lpips.LPIPS(net='alex').cuda()
        self.dists_metric = DISTS().cuda()
        self.fid_metric = FrechetInceptionDistance().cuda()
        self.patch_size = patch_size                # for FID
        self.split_patch_num = split_patch_num      # for FID

        # save pd
        self.details_df = pd.DataFrame()
    
    @torch.no_grad()
    def calculate(self):
        # loop over all images
        for i, (ref_img, recon_img, img_name) in tqdm(enumerate(self.imgloader), total=len(self.imgloader)):
            img_name = img_name[0]
            ref_img = ref_img.to(args.device)
            recon_img = recon_img.to(args.device)

            if self.patch_size != -1:
                update_patch_fid(ref_img, recon_img, fid_metric=self.fid_metric,
                                patch_size=self.patch_size, split_patch_num=self.split_patch_num)

            mse = torch.mean((ref_img - recon_img) ** 2)
            psnr = -10 * torch.log10(mse).item()
            msssim = ms_ssim(ref_img, recon_img, data_range=1.).item()
            lpips_item = self.lpips_metric.forward(ref_img * 2 - 1, recon_img * 2 - 1).item()
            dists_item = self.dists_metric.forward(ref_img, recon_img).item()

            quality_this = {
                'name': img_name,
                'psnr': psnr,
                'msssim': msssim,
                'lpips': lpips_item,
                'dists': dists_item
            }
            quality_this = pd.DataFrame(quality_this, index=[i])
            self.details_df = pd.concat([self.details_df, quality_this])

        # calculate average dict
        avg_dict = self.details_df.mean(numeric_only=True).to_dict()

        # calculate average FID, KID
        if self.patch_size != -1:
            fid = self.fid_metric.compute().item()
            avg_dict['fid'] = float(fid)
            print(f"FID: {float(fid)}")

        # record experiment name
        avg_dict['name'] = self.exp_name

        avg_df = pd.DataFrame([avg_dict])
        return avg_df, self.details_df


@torch.no_grad()
def test(args):
    print(f"Fid patch size: {args.fid_patch_size}")
    print(f"Fid patch num: {args.fid_patch_num}")

    test_opt = test_two_folder(args.recon, args.ref, patch_size=args.fid_patch_size, 
                               split_patch_num=args.fid_patch_num)
    quality_avg, quality_all = test_opt.calculate()

    print("\n=========> Details")
    print(quality_all)

    print("\n=========> Summary")
    print(quality_avg)

    if args.output_name and args.output_path:
        detail_name = "quality_detail_" + args.output_name + ".xlsx"
        detail_file = os.path.join(args.output_path, detail_name)
        quality_all.to_excel(detail_file)

        summary_name = "quality_summary_" + args.output_name + ".xlsx"
        summary_file = os.path.join(args.output_path, summary_name)
        quality_avg.to_excel(summary_file)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str)
    parser.add_argument("--recon", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fid_patch_size", type=int, default=256)
    parser.add_argument("--fid_patch_num", type=int, default=2)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--output_name", type=str, default="")
    args = parser.parse_args()
    test(args)