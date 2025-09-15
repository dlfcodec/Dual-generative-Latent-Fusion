import os
import random as rd
import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F


class ResizeIfSmall(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, img):
        _, height, width = F.get_dimensions(img)
        if height < self.patch_size or width < self.patch_size:
            img = F.resize(img, self.patch_size)
        return img


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            if self.random_crop:
                self.preprocessor = transforms.Compose([
                    ResizeIfSmall(size),
                    transforms.RandomCrop(size),
                    transforms.ToTensor(),
                ])
            else:
                self.preprocessor = transforms.Compose([
                    ResizeIfSmall(size),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                ])
        else:
            self.preprocessor = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.preprocessor(image)
        image = image * 2.0 - 1.0   # [0,1] -> [-1,1]
        image = image.permute(1, 2, 0)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=True)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)



######  new random resize model  #####

class RandomScaleResize(nn.Module):
    def __init__(self, patch_size, resize_prob=0.2):
        super().__init__()
        assert 0.0 <= resize_prob <= 1.0
        self.patch_size = patch_size
        self.resize_prob = resize_prob
        self.resized_crop = transforms.Resize(patch_size)
    
    def forward(self, img):
        _, height, width = F.get_dimensions(img)
        if rd.random() < self.resize_prob:
            img = self.resized_crop(img)
        else:
            if height < self.patch_size or width < self.patch_size:
                img = F.resize(img, self.patch_size)
        return img


class ImagePaths_rdResize(ImagePaths):
    def __init__(self, paths, size=None, random_crop=False, labels=None, p_resize=0.2):
        super().__init__(paths, size, random_crop, labels)

        if self.size is not None and self.size > 0:
            if self.random_crop:
                self.preprocessor = transforms.Compose([
                    RandomScaleResize(size, p_resize),
                    transforms.RandomCrop(size),
                    transforms.ToTensor(),
                ])
            else:
                self.preprocessor = transforms.Compose([
                    ResizeIfSmall(size),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                ])
        else:
            self.preprocessor = transforms.Compose([
                transforms.ToTensor(),
            ])


class CustomTrain_rdResize(CustomBase):
    def __init__(self, size, training_images_list_file, p_resize=0.2):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths_rdResize(paths=paths, size=size, random_crop=True, p_resize=p_resize)



if __name__ == "__main__":
    data_list = "/home/v-naifuxue/datasets/CLIC/clic2020_test.txt"
    crop_size = 1536
    train_set = CustomTrain(crop_size, data_list)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(train_set.data, batch_size=2, shuffle=True)

    for i, img in enumerate(dataloader):
        i = i
        img = img
        pass

