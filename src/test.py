# ---------- import packages ----------
import torch
torch.set_grad_enabled(False)

import argparse
from pathlib import Path
import struct
import yaml
import os, sys, importlib, shutil
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.getcwd())
sys.path.append(f"{os.getcwd()}/../")

from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.utils import save_image, make_grid

from omegaconf import OmegaConf

from taming.modules.losses.quality import Quality_Model
from entropy.compression_model import get_padding_size
from test_scripts.calculate_quality import update_patch_fid, FrechetInceptionDistance


def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))

def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


# ---------- define dataloader ----------
class Test_Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        image = self.transform(image)
        image = image * 2.0 - 1.0
        image_name = os.path.splitext(os.path.split(image_ori)[1])[0]
        return image, image_name

    def __len__(self):
        return len(self.image_path)


@ torch.no_grad()
def test(args):
    # define dataset
    test_set = Test_Dataset(args.dataset_dir)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    
    # define device
    DEVICE = f"cuda:{args.gpu_idx}"
    
    # define model
    config_file = args.base_config
    cfg = load_config(config_file)

    # del original ckpt, and use dummy loss
    cfg.model.params.ckpt_path = args.ckpt_path
    cfg.model.params.ignore_keys = ['epoch_for_strategy', 'lmbda_idx', 'lmbda_list']
    
    out_dir = args.save_dir
    img_dir = os.path.join(out_dir, f"recon")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=False)
    os.makedirs(img_dir, exist_ok=False)
    
    # define output files
    detail_file = os.path.join(out_dir, f"details.csv")
    summary_file = os.path.join(out_dir, f"summary.csv")
    detail_df = pd.DataFrame()
    
    # load model
    model = instantiate_from_config(cfg.model)
    model = model.to(DEVICE).eval()
    model.hybrid_codec.quantize_feat.force_zero_thres = 0.12
    model.hybrid_codec.quantize_feat.update(force=True)
    
    # quality model
    quality_model = Quality_Model().to(DEVICE).eval()
    fid_metric = FrechetInceptionDistance().to(DEVICE)
    print(f"[INFO] fid_patch_size: {args.fid_patch_size}, fid_split_patch_num: {args.fid_split_patch_num}")

    # loop over dataset
    for i, (img, img_name) in tqdm(enumerate(test_loader), total=len(test_loader)):
        img = img.to(DEVICE)
        img_name = img_name[0]

        im_H, im_W = img.shape[2], img.shape[3]

        # padding
        padding_l, padding_r, padding_t, padding_b = get_padding_size(im_H, im_W, p=256)
        img_padded = torch.nn.functional.pad(
            img, (padding_l, padding_r, padding_t, padding_b), mode="replicate",
        )

        # calculate bpp for 256x256 blocks
        bits_titok = (12.0 * 32 / 256 / 256) * img_padded.shape[2] * img_padded.shape[3]
        bpp_titok = bits_titok / (im_H * im_W)
        
        # encoding image
        img_rec_padded, bpp_dict, enc_result = model.encode_decode(img_padded, (im_H, im_W))

        # unpadding
        img_rec = torch.nn.functional.pad(
            img_rec_padded, (-padding_l, -padding_r, -padding_t, -padding_b)
        )
        img_rec = img_rec.clamp(-1.0, 1.0) / 2.0 + 0.5
        img = img / 2.0 + 0.5

        # calculate fid
        update_patch_fid(img, img_rec, 
                         fid_metric=fid_metric, 
                         patch_size=args.fid_patch_size,
                         split_patch_num=args.fid_split_patch_num) 
        
        # record bpp and quality
        quality_this = quality_model(img, img_rec)
        quality_this["name"] = img_name
        quality_this["bpp_titok"] = bpp_titok
        quality_this["bpp_y"] = bpp_dict["h_bpp"]
        quality_this["bpp_titok"] = bpp_dict["z_bpp"]
        quality_this["bpp"] = bpp_dict["total_bpp"]

        # record to dataframe
        quality_this = pd.DataFrame(quality_this, index=[i])
        detail_df = pd.concat([detail_df, quality_this])
    
        # save recon image
        save_image(img_rec, os.path.join(img_dir, f"{img_name}.png"))

        # for save memory
        torch.cuda.empty_cache()
    
    # save detail data frame
    detail_df.to_csv(detail_file)

    # save average frame
    avg_dict = detail_df.mean(numeric_only=True).to_dict()
    avg_dict['fid'] = fid_metric.compute().item()
    pd.DataFrame([avg_dict]).to_csv(summary_file)
    print(avg_dict)

    return None


def init_func():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark  = False
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)


if __name__ == "__main__":
    init_func()
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config', type=str, help='path to base config')
    parser.add_argument('--ckpt_path', type=str, help='path to checkpoint')
    parser.add_argument('--dataset_dir', type=str, help='path to dataset')
    parser.add_argument('--save_dir', type=str, help='path to save results')
    parser.add_argument('--gpu_idx', type=int, default=0, help='gpu index')
    parser.add_argument('--fid_patch_size', type=int, default=256, help='patch size for fid calculation')
    parser.add_argument('--fid_split_patch_num', type=int, default=2, help='split patch number for fid calculation')
    args = parser.parse_args()
    test(args)


