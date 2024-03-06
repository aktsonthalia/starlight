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

held_out_model_wandb_links = wandb_links[HELD_OUT_WANDB_LINKS_KEY]
anchors_wandb_links = wandb_links[ANCHOR_WANDB_LINKS_KEY]
stars_wandb_links = wandb_links[STAR_WANDB_LINKS_KEY]

# plot loss barrier against number of anchors for the star model
print(f"Plotting loss barrier against number of anchors for the star model")

anchor_counts = []
star_held_out_barrier_means = []
star_held_out_barrier_stds = []

for star_wandb_link in stars_wandb_links:

    wandb_run = wandb_api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{star_wandb_link.split('/')[-1]}")
    wandb_config = wandb_run.config
    wandb_config = DotMap(wandb_config)
    anchor_model_wandb_ids = wandb_config.model.anchor_model_wandb_ids
    num_anchors = len(anchor_model_wandb_ids)
    if num_anchors == 1:
        continue

    anchor_counts.append(num_anchors)
    # get all data from the wandb link
    summary = wandb_run.summary
    barriers = summary[HELD_OUT_LOSS_BARRIER_KEY]
    star_held_out_barrier_means.append(mean(barriers))
    star_held_out_barrier_stds.append(stdev(barriers))

plt.errorbar(anchor_counts, star_held_out_barrier_means, yerr=star_held_out_barrier_stds, fmt='o', color=STAR_COLOR, label="Star models")
plt.xlabel("Number of source models")
plt.ylabel("Loss barrier with arbitrary models")
plt.grid()  

# find mean and std of loss barrier with held-out models
anchors_held_out_loss_barrier_means = []

for anchor_wandb_link in anchors_wandb_links:
    wandb_run = wandb_api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{anchor_wandb_link.split('/')[-1]}")
    summary = wandb_run.summary
    barriers = summary[HELD_OUT_LOSS_BARRIER_KEY]   
    anchors_held_out_loss_barrier_means.append(mean(barriers))

anchor_held_out_loss_barriers_mean = mean(anchors_held_out_loss_barrier_means)
anchor_held_out_loss_barriers_std = stdev(anchors_held_out_loss_barrier_means)

# horizontal line at mean loss barrier
plt.axhline(y=anchor_held_out_loss_barriers_mean, color=ANCHOR_COLOR, linestyle='-', label="Regular models")
plt.axhline(y=anchor_held_out_loss_barriers_mean + anchor_held_out_loss_barriers_std, color=ANCHOR_COLOR, linestyle='--')
plt.axhline(y=anchor_held_out_loss_barriers_mean - anchor_held_out_loss_barriers_std, color=ANCHOR_COLOR, linestyle='--')

plt.legend()
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for n, label in enumerate(ax.yaxis.get_ticklabels()):
    if n % 2 != 0:
        label.set_visible(False)
plt.savefig(os.path.join(PLOTS_PATH, f"{config_name}_loss_barrier_vs_num_anchors.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(PLOTS_PATH, "png", f"{config_name}_loss_barrier_vs_num_anchors.png"), format="png")
plt.cla()

# finished plotting loss barrier against number of anchors for the star model
print("Finished plotting loss barrier against number of anchors for the star model")

def interp_with_held_out_anchors(wandb_ids):
    
    base_run = wandb_api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{wandb_ids[0]}")
    held_out_anchor_ids = base_run.config["eval"]["held_out_anchors"]

    loss_lists = []
    acc_lists = []

    for wandb_id in wandb_ids:

        wandb_run = wandb_api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{wandb_id}")
        for held_out_anchor_id in held_out_anchor_ids:
            artifacts = wandb_run.logged_artifacts()
            held_out_table_name = f"run-{wandb_id}-interp_with_held_out{held_out_anchor_id}_loss_table:v0"
            held_out_table = [artifact for artifact in artifacts if artifact.name == held_out_table_name][0]
            download_path = os.environ["SCRATCH"] + f"/{held_out_table_name}"
            held_out_table.download(download_path)
            files = os.listdir(download_path)
            assert len(files) == 1
            assert files[0].endswith(".json")

            # read the json file
            with open(download_path + f"/{files[0]}") as f:
                held_out_table = json.load(f)
            
            held_out_df = pd.DataFrame(columns=held_out_table["columns"], data=held_out_table["data"])

            t = held_out_df["t"]
            try:
                loss_lists.append(held_out_df["test_loss"])
                acc_lists.append([x*100 for x in held_out_df["test_acc"]])
            except KeyError:
                assert config_name == "cifar10_resnet18"
                loss_lists.append(held_out_df["val_loss"])
                acc_lists.append([x*100 for x in held_out_df["val_acc"]])

    print(f"loss_lists: {loss_lists}")
    print(f"acc lists: {acc_lists}")
    # find the mean and std of the loss and acc lists
    loss_means = [mean([x[i] for x in loss_lists]) for i in range(len(t))]
    loss_stds = [stdev([x[i] for x in loss_lists]) for i in range(len(t))]
    acc_means = [mean([x[i] for x in acc_lists]) for i in range(len(t))]
    acc_stds = [stdev([x[i] for x in acc_lists]) for i in range(len(t))]

    return t, loss_means, loss_stds, acc_means, acc_stds

def style_plot(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % 2 != 0:
            label.set_visible(False)

anchor_wandb_ids = [link.split("/")[-1] for link in random.sample(anchors_wandb_links, NUM_ANCHORS_TO_SAMPLE)]
t, anchor_loss_means, anchor_loss_stds, anchor_acc_means, anchor_acc_stds = interp_with_held_out_anchors(anchor_wandb_ids)
star_wandb_id = stars_wandb_links[-1].split("/")[-1]
t, star_loss_means, star_loss_stds, star_acc_means, star_acc_stds = interp_with_held_out_anchors([star_wandb_id])

last_index = len(t) - 1
# breakpoint()
plt.plot(t, anchor_loss_means, label="regular-regular", color=ANCHOR_COLOR)
plt.plot(t, star_loss_means, label="star-heldout", color=STAR_COLOR)

anchor_loss_upper_bounds = [anchor_loss_means[i] + anchor_loss_stds[i] for i in range(len(anchor_loss_means))]
anchor_loss_lower_bounds = [anchor_loss_means[i] - anchor_loss_stds[i] for i in range(len(anchor_loss_means))]
star_loss_upper_bounds = [star_loss_means[i] + star_loss_stds[i] for i in range(len(star_loss_means))]
star_loss_lower_bounds = [star_loss_means[i] - star_loss_stds[i] for i in range(len(star_loss_means))]
# t = list(t)
# t = np.array(t, dtype=np.float32)
anchor_loss_lower_bounds = np.array(anchor_loss_lower_bounds, dtype=np.float32)
anchor_loss_upper_bounds = np.array(anchor_loss_upper_bounds, dtype=np.float32)

plt.fill_between(t, anchor_loss_upper_bounds, anchor_loss_lower_bounds, color=ANCHOR_COLOR, alpha=0.2)
plt.fill_between(t, star_loss_upper_bounds, star_loss_lower_bounds, color=STAR_COLOR, alpha=0.2)
plt.xlabel("t")
plt.ylabel("test loss")
plt.grid()
plt.legend(loc="center left")
ax = plt.gca()
style_plot(ax)
# plt.subplots_adjust(left=0.2)
# plt.subplots_adjust(bottom=0.2)
plt.savefig(os.path.join(PLOTS_PATH, f"{config_name}_held_out_loss.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(PLOTS_PATH, "png", f"{config_name}_held_out_loss.png"), format="png")
plt.cla()

plt.plot(t, anchor_acc_means, label="regular-regular", color=ANCHOR_COLOR)
plt.plot(t, star_acc_means, label="star-heldout", color=STAR_COLOR)

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
style_plot(ax)
# plt.subplots_adjust(left=0.2)
# plt.subplots_adjust(bottom=0.2)
plt.savefig(os.path.join(PLOTS_PATH, f"{config_name}_held_out_acc.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(PLOTS_PATH, "png", f"{config_name}_held_out_acc.png"), format="png")
plt.cla()

dbx.upload_file(os.path.join(PLOTS_PATH, f"{config_name}_held_out_acc.svg"), os.path.join(DROPBOX_REMOTE_FOLDER, f"{config_name}_held_out_acc.svg"))
dbx.upload_file(os.path.join(PLOTS_PATH, f"{config_name}_held_out_loss.svg"), os.path.join(DROPBOX_REMOTE_FOLDER, f"{config_name}_held_out_loss.svg"))
dbx.upload_file(os.path.join(PLOTS_PATH, f"{config_name}_loss_barrier_vs_num_anchors.svg"), os.path.join(DROPBOX_REMOTE_FOLDER, f"{config_name}_loss_barrier_vs_num_anchors.svg"))