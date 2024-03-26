import sys

import torchvision
from torch.optim import AdamW, Adam, SGD

from .densenet import densenet_cifar as densenet_cifar
from .resnet_cifar_std import ResNet18 as resnet18_cifar
from .vgg import vgg11
from .wide_resnet import Wide_ResNet

sys.path.append("..")

models_dict = {
    "resnet18_cifar": resnet18_cifar,
    "resnet18_torch": torchvision.models.resnet18,
    "resnet50_torch": torchvision.models.resnet50,
    "densenet_cifar": densenet_cifar,
    "vgg11": vgg11,
    "wrn": Wide_ResNet
}

optimizers_dict = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW
}