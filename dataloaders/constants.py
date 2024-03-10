import numpy as np
import os 

DATASETS_PATH = os.path.join(os.environ["WORK"], "datasets")
DATASET_SPLIT_SEED = 42
# to avoid mistakes during evaluation, we always use a fixed batch size which is a divisor of the number of samples
# just an extra precaution
VAL_BATCH_SIZE = 100

IMAGENET_MEANS = np.array([0.485, 0.456, 0.406])
IMAGENET_STDS = np.array([0.229, 0.224, 0.225])
CIFAR10_NUM_CLASSES = 10
CIFAR100_NUM_CLASSES = 100
CIFAR10_PATH = os.path.join(os.environ["WORK"], "datasets", "cifar10")
CIFAR100_PATH = os.path.join(os.environ["WORK"], "datasets", "cifar100")
CIFAR10_MEANS = IMAGENET_MEANS
CIFAR10_STDS = IMAGENET_STDS
CIFAR100_MEANS = IMAGENET_MEANS
CIFAR100_STDS = IMAGENET_STDS

IMAGENET_PATH = "/mnt/qb/datasets/ImageNet-ffcv"
IMAGENET_TRAIN_FFCV = "train_500_0.50_90.ffcv"
IMAGENET_VAL_FFCV = "val_500_0.50_90.ffcv"

IMAGENET_SIZE = (224, 224)


