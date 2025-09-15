import argparse, os, sys, datetime
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint, Callback, LearningRateMonitor,
)
from pytorch_lightning.utilities import rank_zero_only

from taming.util import instantiate_from_config


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    add = parser.add_argument
    add("-o", "--outdir", type=str, default="logs", help="ckpt and log output dir")
    add("-n", "--name", type=str, default="", help="logdir postfix")
    add("-b", "--base", type=str, required=True, help="base config paths")
    add("-d", "--debug", action="store_true", default=False, help="debug model")
    add("-s", "--seed", type=int, default=114514, help="random seed")
    add("--gpus", type=str, default="", help="gpu device(s), like 0,1,2,3")
    add("--log_rate", type=int, default=50, help="log rate")
    add("--resume", type=str, default="", help="resume ckpt path")
    add("--save_top_k", type=int, default=3, help="save top k models")
    return parser


class SetupCallback(Callback):
    def __init__(self, now, logdir, cfgdir, config, lightning_config):
        super().__init__()
        self.now = now
        self.logdir = logdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # save configs
            os.makedirs(self.cfgdir, exist_ok=True)
            OmegaConf.save(self.config, os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}), 
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))


class ImageLogger(Callback):
    def __init__(self, imgdir, batch_frequency, max_images=4, clamp=True, 
                 increase_log_steps=True, out_img_file=True):
        super().__init__()
        self.imgdir = imgdir
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.out_img_file = out_img_file
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp


    @rank_zero_only
    def log_local(self, save_dir, split, images, batch_idx, pl_module):
        root = os.path.join(save_dir, split)
        global_step = pl_module.global_step
        current_epoch = pl_module.current_epoch
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            pl_module.logger.experiment.add_image(f"{split}/{k}", grid, global_step)

            if self.out_img_file:
                grid_out = grid.transpose(0,1).transpose(1,2).squeeze(-1)
                grid_out = grid_out.numpy()
                grid_out = (grid_out*255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k,
                    global_step,
                    current_epoch,
                    batch_idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grid_out).save(path)


    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(self.imgdir, split, images, batch_idx, pl_module)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")



if __name__ == "__main__":

    # parse args
    parser = get_parser()
    args, unknown = parser.parse_known_args()

    # create the save folder
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if args.name:
        name = "_"+args.name
    elif args.base:
        cfg_fname = os.path.split(args.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_"+cfg_name
    else:
        name = ""
    nowname = now+name
    logdir = os.path.join(args.outdir, nowname)
    imgdir = os.path.join(logdir, "images")
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    # set seed
    if args.seed < 0:
        args.seed = np.random.randint(0, 2**31 - 1)
    seed_everything(args.seed)

    # load base configs
    config = OmegaConf.load(args.base)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    # set gpus
    if args.gpus:
        trainer_config['accelerator'] = 'gpu'
        trainer_config['devices'] = [int(g) for g in args.gpus.split(',') if g]
        if len(trainer_config['devices']) != 1:
            trainer_config['strategy'] = 'ddp_find_unused_parameters_true'
    else:
        trainer_config['accelerator'] = 'cpu'

    # put config back to the lightning file
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # instaniate model
    model = instantiate_from_config(config.model)

    # trainer and callbacks
    trainer_kwargs = dict(trainer_config)

    # define logger
    trainer_kwargs["logger"] = TensorBoardLogger(name="logs", save_dir=logdir)
    trainer_kwargs["log_every_n_steps"] = args.log_rate

    # define callbacks
    trainer_kwargs["callbacks"] = []

    # - define model checkpoint
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        monitor_name = model.monitor
    else:
        monitor_name = ""
    
    trainer_kwargs["callbacks"].append(
        ModelCheckpoint(
            mode="min",
            save_top_k=args.save_top_k,
            save_last=True,
            dirpath=ckptdir, 
            filename="{epoch}-{saved_loss:.4f}", 
            monitor=monitor_name if monitor_name else None,
            verbose=True,
            save_weights_only=False,
            every_n_epochs=1,
        )
    )

    # - define setup callback
    trainer_kwargs["callbacks"].append(
        SetupCallback(
            now=now,
            logdir=logdir,
            cfgdir=cfgdir,
            config=config,
            lightning_config=lightning_config,
        )
    )

    # - define image logger callback
    trainer_kwargs["callbacks"].append(
        ImageLogger(
            imgdir=imgdir,
            batch_frequency=2000,
            max_images=4,
            clamp=True,
            out_img_file=False
        )
    )

    # - learning rate logger callback
    trainer_kwargs["callbacks"].append(
        LearningRateMonitor(
            logging_interval="step",
        )
    )

    # data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    # define trainer
    trainer = Trainer(**trainer_kwargs)

    # run
    if args.resume:
        trainer.fit(model, 
                    train_dataloaders=data._train_dataloader(),
                    val_dataloaders=data._val_dataloader(),
                    ckpt_path=args.resume)
    else:
        trainer.fit(model, 
                    train_dataloaders=data._train_dataloader(),
                    val_dataloaders=data._val_dataloader())
