import os

WANDB_ENTITY = "mode-connect"
WANDB_PROJECT = "star-domain"
PROJECT_PATH = os.path.join(os.environ["HOME"], "starlight")
RESULTS_DIR = os.path.join(PROJECT_PATH, "results")

WANDB_LINKS_DIR = os.path.join(RESULTS_DIR, "wandb_links")
PLOTS_PATH = os.path.join(RESULTS_DIR, "figures")

WORKSHEET_NAMES = {
    "star_model_hypothesis": {
        "cifar10_resnet18": {
            "anchors": "resnet18_cifar10_anchors",
            "stars": "resnet18_cifar10_stars"
        },
        "cifar100_resnet18": {
            "anchors": "resnet18_cifar100_anchors",
            "stars": "resnet18_cifar100_stars"
        },
    }
}

HELD_OUT_WANDB_LINKS_KEY = "held_out"
ANCHOR_WANDB_LINKS_KEY = "anchors"
STAR_WANDB_LINKS_KEY = "stars"
HELD_OUT_LOSS_BARRIER_KEY = "all_barriers_loss_held_out"

DROPBOX_LOCAL_FOLDER = PLOTS_PATH
DROPBOX_REMOTE_FOLDER = "/Apps/Overleaf/Star-model/figures"

import matplotlib.pyplot as plt

plt.tight_layout()
# Set global Matplotlib parameters
plt.rcParams['font.size'] = 16
# ylabel size 
# plt.rcParams['axes.labelsize'] = 14
# plt.rcParams['axes.labelsize'] = 24
# plt.rcParams['axes.titlesize'] = 24
# plt.rcParams['xtick.labelsize'] = 24
# plt.rcParams['ytick.labelsize'] = 
# plt.rcParams['legend.fontsize'] = 24
plt.rcParams['svg.fonttype'] = 'none'

ANCHOR_COLOR = "blue"
STAR_COLOR = "red"