import sys

import torchvision
from torch.optim import AdamW, Adam, SGD

from .densenet import densenet_cifar as densenet_cifar
from .resnet_cifar_std import ResNet18 as resnet18_cifar

sys.path.append("..")

models_dict = {
    "resnet18_cifar": resnet18_cifar,
    "resnet18_torch": torchvision.models.resnet18,
    "densenet_cifar": densenet_cifar
}

optimizers_dict = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW
}