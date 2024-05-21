import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .constants import IMAGENET_PATH, IMAGENET_TRAIN_DIR, IMAGENET_VAL_DIR, IMAGENET_MEANS, IMAGENET_STDS

IMAGENET_MEANS_SCALED = IMAGENET_MEANS * 255
IMAGENET_STDS_SCALED = IMAGENET_STDS * 255

def load_imagenet1k_without_ffcv(
    batch_size, 
    num_workers=8, 
    img_size=224,
    use_augmentation=True
):
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS),
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(IMAGENET_PATH, IMAGENET_TRAIN_DIR), 
        transform=train_transforms if use_augmentation else val_transforms
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(IMAGENET_PATH, IMAGENET_VAL_DIR), 
        transform=val_transforms
    )

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_dl = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dl, val_dl, val_dl