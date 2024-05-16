# from .ag_news import load_ag_news
from .cifar10 import load_cifar10
from .cifar100 import load_cifar100
from .imagenet import load_imagenet1k
from .mnist import load_mnist

datasets_dict = {
    # "ag_news": load_ag_news,
    "cifar10": load_cifar10,
    "cifar100": load_cifar100,
    "imagenet1k": load_imagenet1k,
    "mnist": load_mnist,
}
