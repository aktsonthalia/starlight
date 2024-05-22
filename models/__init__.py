import sys
from functools import partial   

import torchvision
from torch.optim import AdamW, Adam, SGD

from .densenet import densenet_cifar as densenet_cifar
from .mlp import MLP
from .resnet_cifar_std import ResNet18 as resnet18_cifar
from .vgg import vgg11, vgg19, vgg11_bn, vgg19_bn
from .text_classifier import SimpleTextClassifier, TextCNN
from .wide_resnet import Wide_ResNet

models_dict = {
    "mlp_mnist": partial(MLP, input_dim=784),
    "resnet18_cifar": resnet18_cifar,
    "resnet18_torch": torchvision.models.resnet18,
    "resnet50_torch": torchvision.models.resnet50,
    "densenet_cifar": densenet_cifar,
    "vgg11": vgg11,
    "vgg19": vgg19,
    "vgg11_bn": vgg11_bn,
    "vgg19_bn": vgg19_bn,
    "wrn": Wide_ResNet,
    "simple_text_classifier": SimpleTextClassifier,
    "text_cnn": TextCNN
}

optimizers_dict = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW
}