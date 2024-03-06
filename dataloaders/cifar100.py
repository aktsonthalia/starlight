import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision import transforms

from .constants import *
from .utils import RandomApplyOne, IdentityTransform


def load_cifar100(
    batch_size=256, 
    num_workers=8, 
    img_size=32, 
    normalize=True,
    resize=False, 
    horizontal_flip=False,
    vertical_flip=False,
    random_crop_resize=False,
    random_resize_crop=False,
    color_jitter=False,
    rotation_range=0,  
    pad_random_crop=False,
    random_one_aug=False,   
    train_set_fraction=1.0,
    return_ds=False,
):
    # define validation transform
    val_transform = transforms.Compose(
        [
            transforms.Resize(img_size) if resize else IdentityTransform(),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR100_MEANS, std=CIFAR100_STDS) if normalize else IdentityTransform()
        ]
    )

    # define test dataloader
    test_ds = torchvision.datasets.CIFAR100(
        root=CIFAR100_PATH,
        train=False,
        download=True,
        transform=val_transform,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # define train transform
    augmentations = []
    if horizontal_flip:
        augmentations.append(transforms.RandomHorizontalFlip())
    if pad_random_crop:
        augmentations.append(transforms.RandomCrop(size=img_size, padding=4))
    if random_crop_resize:
        augmentations.append(transforms.RandomResizedCrop(size=img_size, scale=(0.08, 1)))
    if vertical_flip:
        augmentations.append(transforms.RandomVerticalFlip())
    if rotation_range > 0:
        augmentations.append(transforms.RandomRotation(rotation_range))
    if color_jitter:
        augmentations.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    if random_one_aug:
        augmentations = [RandomApplyOne(augmentations)]

    train_transform = transforms.Compose(
        [
            transforms.Resize(img_size) if resize else IdentityTransform(),
        ] + augmentations + [
            transforms.ToTensor(),  
            transforms.Normalize(mean=CIFAR100_MEANS, std=CIFAR100_STDS) if normalize else IdentityTransform()
        ]
    )

    # define train and validation dataloaders
    train_ds = torchvision.datasets.CIFAR100(
        root=CIFAR100_PATH,
        train=True,
        download=True,
        transform=train_transform,
    )

    val_ds = torchvision.datasets.CIFAR100(
        root=CIFAR100_PATH,
        train=True,
        download=True,
        transform=val_transform,
    )

    # randomly split train_ds into train_ds and val_ds
    # use a fixed seed for reproducibility
    indices = torch.randperm(
        len(train_ds), generator=torch.Generator().manual_seed(DATASET_SPLIT_SEED)
    ).tolist()
    split_point = int(len(train_ds) * 0.9)
    train_set_size = int(split_point * train_set_fraction)
    train_ds = torch.utils.data.Subset(train_ds, indices[:train_set_size])
    val_ds = torch.utils.data.Subset(val_ds, indices[split_point:])

    print(f"Using {len(train_ds)} images for training")
    print(f"Using {len(val_ds)} images for validation")
    
    if return_ds:
        return train_ds, val_ds
    
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Using train transform {train_transform}")

    # assert batch size of val and test dl can divide the number of samples
    assert len(test_ds) % test_dl.batch_size == 0
    assert len(val_ds) % val_dl.batch_size == 0
    
    train_dl.img_size = img_size
    test_dl.num_classes = CIFAR100_NUM_CLASSES
    return train_dl, val_dl, test_dl