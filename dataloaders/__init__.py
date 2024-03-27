from .cifar10 import load_cifar10
from .cifar100 import load_cifar100
from .imagenet import load_imagenet1k
from .sst import load_sst

datasets_dict = {
    "cifar10": load_cifar10,
    "cifar100": load_cifar100,
    "imagenet1k": load_imagenet1k,
    "sst": load_sst
}
