# from .ag_news import load_ag_news
from .cifar10 import load_cifar10
from .cifar100 import load_cifar100
from .imagenet import load_imagenet1k
from .imagenet_without_ffcv import load_imagenet1k_without_ffcv
from .mnist import load_mnist
from .ag_news import load_ag_news

datasets_dict = {
    # "ag_news": load_ag_news,
    "cifar10": load_cifar10,
    "cifar100": load_cifar100,
    "imagenet1k": load_imagenet1k_without_ffcv,
    "mnist": load_mnist,
    "ag_news": load_ag_news,
}
