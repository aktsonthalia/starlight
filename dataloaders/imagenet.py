import os
import numpy as np
import torch

try:
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import (
        ToTensor,
        ToDevice, 
        ToTorchImage, 
        Squeeze,
        NormalizeImage,
        RandomHorizontalFlip,
    )
    from ffcv.fields.decoders import (
        IntDecoder, 
        RandomResizedCropRGBImageDecoder, 
        CenterCropRGBImageDecoder
    )
except:
    pass

from .constants import *


IMAGENET_MEANS_SCALED = IMAGENET_MEANS * 255
IMAGENET_STDS_SCALED = IMAGENET_STDS * 255

def load_imagenet1k(batch_size, num_workers, img_size=224, rand_aug_magnitude=10, rand_aug_num_ops=2):
    
    image_pipeline_train = [
        RandomResizedCropRGBImageDecoder((img_size, img_size)),
        RandomHorizontalFlip(),
        ToTensor(),         
        ToDevice(torch.device("cuda"), non_blocking=True),
        ToTorchImage(),
        # RandAugment(magnitude=rand_aug_magnitude, num_ops=rand_aug_num_ops),
        # ConvertImageDtype(torch.float32),
        NormalizeImage(IMAGENET_MEANS_SCALED, IMAGENET_STDS_SCALED, np.float32),
    ]
    label_pipeline_train = [
        IntDecoder(), 
        ToTensor(), 
        Squeeze(), 
        ToDevice(torch.device("cuda"), non_blocking=True)
    ]
    train_dl = Loader(
        os.path.join(IMAGENET_PATH, IMAGENET_TRAIN_FFCV), 
        batch_size=batch_size, 
        os_cache=1, 
        num_workers=num_workers, 
        order=OrderOption.RANDOM,
        pipelines={
            'image': image_pipeline_train, 
            'label': label_pipeline_train
        },
        distributed=False, 
        drop_last=True
    )
    
    image_pipeline_val = [
        CenterCropRGBImageDecoder((224, 224), ratio=224/256),
        ToTensor(), 
        ToDevice(torch.device("cuda"), non_blocking=True),
        ToTorchImage(), 
        NormalizeImage(IMAGENET_MEANS_SCALED, IMAGENET_STDS_SCALED, np.float32)
    ]
    label_pipeline_val = [
        IntDecoder(), 
        ToTensor(), 
        Squeeze(), 
        ToDevice(torch.device("cuda"), non_blocking=True)
    ]
    val_dl = Loader(
        os.path.join(IMAGENET_PATH, IMAGENET_VAL_FFCV), 
        batch_size=batch_size, 
        os_cache=1, 
        num_workers=num_workers, 
        order=OrderOption.SEQUENTIAL, 
        pipelines={
            'image': image_pipeline_val, 
            'label': label_pipeline_val
        }, 
        distributed=False, 
        drop_last=True 
    )

    # TODO: temporary hack; get proper val set later
    train_dl.img_size = img_size
    return train_dl, val_dl, val_dl