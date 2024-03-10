from constants import *
import sys
sys.path.append(PROJECT_PATH)

import argparse
import json
import numpy as np
import random
import pandas as pd
import wandb
import yaml

from make_star_model_hypothesis_plots import interp_with_held_out_anchors
from dotmap import DotMap
from statistics import mean, stdev 

from constants import *
from utils import DropboxSync
dbx = DropboxSync(os.environ["DAT"])


TEXT_FONT_SIZE = 10

wandb_api = wandb.Api()

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str)
args = parser.parse_args()

WANDB_LINKS_DIR = os.path.join(RESULTS_DIR, "wandb_links")
config_name = args.config
yaml_path = os.path.join(WANDB_LINKS_DIR, f"{config_name}.yaml")

if "imagenet" in config_name:
    NUM_ANCHORS_TO_SAMPLE = 3
else:
    NUM_ANCHORS_TO_SAMPLE = 5

with open(yaml_path) as f:
    wandb_links = yaml.safe_load(f)

anchors_wandb_links = wandb_links[ANCHOR_WANDB_LINKS_KEY]
stars_wandb_links = wandb_links[STAR_WANDB_LINKS_KEY]

def interp_with_anchors(wandb_ids):
    
    loss_lists = []
    acc_lists = []

    for wandb_id in wandb_ids:

        wandb_run = wandb_api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{wandb_id}")
        artifacts = wandb_run.logged_artifacts()
        anchor_interp_artifacts = [artifact for artifact in artifacts if "startraininganchor" in artifact.name]
        # assert len(anchor_interp_artifacts) == 5, print(f"len(anchor_interp_artifacts): {len(anchor_interp_artifacts)}")

        for artifact in anchor_interp_artifacts:
            download_path = os.path.join(os.environ["SCRATCH"], artifact.name)
            artifact.download(download_path)
            files = os.listdir(download_path)
            assert len(files) == 1
            assert files[0].endswith(".json")

            # read the json file
            with open(download_path + f"/{files[0]}") as f:
                interp_table = json.load(f)
            
            interp_df = pd.DataFrame(columns=interp_table["columns"], data=interp_table["data"])

            t = interp_df["t"]
            try:
                loss_lists.append(interp_df["test_loss"])
                acc_lists.append([x*100 for x in interp_df["test_acc"]])
            except KeyError:
                assert config_name == "cifar10_resnet18"
                loss_lists.append(interp_df["val_loss"])
                acc_lists.append([x*100 for x in interp_df["val_acc"]])

    print(f"loss_lists: {loss_lists}")
    print(f"acc lists: {acc_lists}")
    # find the mean and std of the loss and acc lists
    loss_means = [mean([x[i] for x in loss_lists]) for i in range(len(t))]
    loss_stds = [stdev([x[i] for x in loss_lists]) for i in range(len(t))]
    acc_means = [mean([x[i] for x in acc_lists]) for i in range(len(t))]
    acc_stds = [stdev([x[i] for x in acc_lists]) for i in range(len(t))]

    return t, loss_means, loss_stds, acc_means, acc_stds

anchor_wandb_ids = [link.split("/")[-1] for link in random.sample(anchors_wandb_links, NUM_ANCHORS_TO_SAMPLE)]
t, anchor_loss_means, anchor_loss_stds, anchor_acc_means, anchor_acc_stds = interp_with_held_out_anchors(anchor_wandb_ids)
star_wandb_id = stars_wandb_links[-1].split("/")[-1]
t, star_loss_means, star_loss_stds, star_acc_means, star_acc_stds = interp_with_anchors([star_wandb_id])

last_index = len(t) - 1
plt.plot(t, anchor_loss_means, label="regular-regular", color=ANCHOR_COLOR)
plt.plot(t, star_loss_means, label="star-source", color=STAR_COLOR)

anchor_loss_upper_bounds = [anchor_loss_means[i] + anchor_loss_stds[i] for i in range(len(anchor_loss_means))]
anchor_loss_lower_bounds = [anchor_loss_means[i] - anchor_loss_stds[i] for i in range(len(anchor_loss_means))]
star_loss_upper_bounds = [star_loss_means[i] + star_loss_stds[i] for i in range(len(star_loss_means))]
star_loss_lower_bounds = [star_loss_means[i] - star_loss_stds[i] for i in range(len(star_loss_means))]

anchor_loss_lower_bounds = np.array(anchor_loss_lower_bounds, dtype=np.float32)
anchor_loss_upper_bounds = np.array(anchor_loss_upper_bounds, dtype=np.float32)

plt.fill_between(t, anchor_loss_upper_bounds, anchor_loss_lower_bounds, color=ANCHOR_COLOR, alpha=0.2)
plt.fill_between(t, star_loss_upper_bounds, star_loss_lower_bounds, color=STAR_COLOR, alpha=0.2)
plt.xlabel("t")
plt.ylabel("test loss")
plt.grid()
plt.legend(loc="center left")
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for n, label in enumerate(ax.yaxis.get_ticklabels()):
    if n % 2 != 0:
        label.set_visible(False)
plt.savefig(os.path.join(PLOTS_PATH, f"{config_name}_anchor_loss.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(PLOTS_PATH, "png", f"{config_name}_anchor_loss.png"), format="png")
# breakpoint()
plt.cla()

plt.plot(t, anchor_acc_means, label="regular-regular", color=ANCHOR_COLOR)
plt.plot(t, star_acc_means, label="star-source", color=STAR_COLOR)

anchor_acc_upper_bounds = [anchor_acc_means[i] + anchor_acc_stds[i] for i in range(len(anchor_acc_means))]
anchor_acc_lower_bounds = [anchor_acc_means[i] - anchor_acc_stds[i] for i in range(len(anchor_acc_means))]
star_acc_upper_bounds = [star_acc_means[i] + star_acc_stds[i] for i in range(len(star_acc_means))]
star_acc_lower_bounds = [star_acc_means[i] - star_acc_stds[i] for i in range(len(star_acc_means))]
plt.fill_between(t, anchor_acc_upper_bounds, anchor_acc_lower_bounds, color=ANCHOR_COLOR, alpha=0.3)
plt.fill_between(t, star_acc_upper_bounds, star_acc_lower_bounds, color=STAR_COLOR, alpha=0.3)
plt.xlabel("t")
plt.ylabel("test accuracy")
plt.grid()
plt.legend(loc="center left")
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for n, label in enumerate(ax.yaxis.get_ticklabels()):
    if n % 2 != 0:
        label.set_visible(False)
plt.savefig(
    os.path.join(
        PLOTS_PATH, 
        f"{config_name}_anchor_acc.svg"
    ), 
    format="svg",
    bbox_inches="tight"
)
plt.savefig(
    os.path.join(
        PLOTS_PATH, 
        "png", 
        f"{config_name}_anchor_acc.png"
    ), 
    format="png"
)
plt.cla()

dbx.upload_file(
    os.path.join(PLOTS_PATH, f"{config_name}_anchor_acc.svg"), 
    os.path.join(DROPBOX_REMOTE_FOLDER, f"{config_name}_anchor_acc.svg")
)
dbx.upload_file(
    os.path.join(PLOTS_PATH, f"{config_name}_anchor_loss.svg"), 
    os.path.join(DROPBOX_REMOTE_FOLDER, f"{config_name}_anchor_loss.svg")
)