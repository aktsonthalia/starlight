import numpy as np
import os 

try:
    DATASETS_PATH = os.path.join(os.environ["WORK"], "datasets") # set this to your own datasets path
except KeyError:
    DATASETS_PATH = "datasets"
    os.makedirs(DATASETS_PATH, exist_ok=True)
    
DATASET_SPLIT_SEED = 42
# to avoid mistakes during evaluation, we always use a fixed batch size which is a divisor of the number of samples
# just an extra precaution
VAL_BATCH_SIZE = 100

IMAGENET_MEANS = np.array([0.485, 0.456, 0.406])
IMAGENET_STDS = np.array([0.229, 0.224, 0.225])
CIFAR10_NUM_CLASSES = 10
CIFAR100_NUM_CLASSES = 100
CIFAR10_PATH = os.path.join(DATASETS_PATH, "cifar10")
CIFAR100_PATH = os.path.join(DATASETS_PATH, "cifar100")

IMAGENET_PATH = "/mnt/qb/datasets/ImageNet-ffcv"
IMAGENET_TRAIN_FFCV = "train_500_0.50_90.ffcv"
IMAGENET_VAL_FFCV = "val_500_0.50_90.ffcv"

CIFAR10_MEANS = IMAGENET_MEANS
CIFAR10_STDS = IMAGENET_STDS
CIFAR100_MEANS = IMAGENET_MEANS
CIFAR100_STDS = IMAGENET_STDS

IMAGENET_SIZE = (224, 224)

AG_NEWS_PATH = os.path.join(DATASETS_PATH, "ag_news")
SST_PATH = os.path.join(DATASETS_PATH, "sst")

MNIST_PATH = os.path.join(DATASETS_PATH, "mnist")
MNIST_MEANS = np.array([0.1307])
MNIST_STDS = np.array([0.3081])
MNIST_NUM_CLASSES = 10