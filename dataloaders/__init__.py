from .cifar10 import load_cifar10
from .cifar100 import load_cifar100
from .imagenet import load_imagenet1k
from .ag_news import load_ag_news

datasets_dict = {
    "cifar10": load_cifar10,
    "cifar100": load_cifar100,
    "imagenet1k": load_imagenet1k,
    "ag_news": load_ag_news,
}
