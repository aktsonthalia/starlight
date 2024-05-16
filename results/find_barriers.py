import sys
sys.path.append("..")   

from functools import partial

import argparse
import yaml
import os
import random
import time
import torch
import wandb 
from dotmap import DotMap
from utils import (
    setup_model, 
    make_interpolation_plot, 
    load_model_from_wandb_id,
    match_weights
)
from dataloaders import datasets_dict

ENTITY = "mode-connect"
PROJECT = "star-domain"
INTERPOLATION_NUM_POINTS = 11

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help="dataset")
parser.add_argument('--input_file', type=str, required=True, help="input file")
parser.add_argument('--model', type=str, required=True, help="model")
parser.add_argument('--output_dir', type=str, required=True, help="output file")
parser.add_argument('--setting', type=str, required=True, help="setting")
args = parser.parse_args()

with open(args.input_file, 'r') as f:
    all_links = yaml.safe_load(f)

# now search for model, dataset, setting combination
count = 0
for obj in all_links:
    if obj['model'] == args.model and obj['dataset'] == args.dataset and obj['setting'] == args.setting:
        links_to_use = obj
        count += 1
assert count == 1, f"Found {count} matches for model {args.model}, dataset {args.dataset}, setting {args.setting}"

result = {
    "model": args.model,
    "dataset": args.dataset,
    "setting": args.setting,
    "star_held_out": [],
    "anchor_held_out": []
}

# get config
api = wandb.Api()
lmw = partial(load_model_from_wandb_id, entity=ENTITY, project=PROJECT)
run = api.run(f"{ENTITY}/{PROJECT}/{links_to_use['held_out'][0]}")
config = DotMap(run.config)

# setup dataset
config.dataset.settings.horizontal_flip = False
train_dl, val_dl, test_dl = datasets_dict[config.dataset.name](
    batch_size=config.dataset.settings.batch_size,
    normalize=config.dataset.settings.normalize,
)

# setup model pairs

model_id_pairs = {
    "star_held_out": [],
    "anchor_held_out": []
}

# reduce number of held-outs and anchors to 3 each 
random.seed(42)
random.shuffle(links_to_use['held_out'])
random.shuffle(links_to_use['anchors'])
links_to_use['held_out'] = links_to_use['held_out'][:3]
links_to_use['anchors'] = links_to_use['anchors'][:3]

for held_out_link in links_to_use['held_out']:
    held_out_id = held_out_link.split('/')[-1]
    for star_link in links_to_use['stars']:
        star_id = star_link.split('/')[-1]
        model_id_pairs["star_held_out"].append((star_id, held_out_id))
    for anchor_link in links_to_use['anchors']:
        anchor_id = anchor_link.split('/')[-1]
        model_id_pairs["anchor_held_out"].append((anchor_id, held_out_id))

for anchor_link in links_to_use['anchors']:
    anchor_id = anchor_link.split('/')[-1]
    for star_link in links_to_use['stars']:
        star_id = star_link.split('/')[-1]
        model_id_pairs["star_anchor"].append((star_id, anchor_id))

for model_pair_type, model_pairs in model_id_pairs.items():

    print(f"Processing {model_pair_type} models")

    for model1_id, model2_id in model_pairs:

        start_time = time.time()

        print(f"Setting up {model1_id} and {model2_id}")
        model1 = setup_model(config)
        model2 = setup_model(config)

        print(f"Loading {model1_id} and {model2_id}")
        model1.load_state_dict(lmw(wandb_id=model1_id))
        model2.load_state_dict(lmw(wandb_id=model2_id))

        print(f"Matching weights between {model1_id} and {model2_id}")
        model2 = match_weights(
            model1, 
            model2, 
            train_dl, 
            recalculate_batch_statistics=True
        )

        print(f"Making interpolation plot between {model1_id} and {model2_id}")
        temp = make_interpolation_plot(
            model1=model1,
            model2=model2,
            num_points=INTERPOLATION_NUM_POINTS,
            logger=None,
            plot_title=f"Interpolation between {model1_id} and {model2_id}",
            train_dl=train_dl,
            test_dl=test_dl,
            verbose=True,
        )

        temp["model1_id"] = model1_id
        temp["model2_id"] = model2_id

        result[model_pair_type].append(temp)

        print(f"Time taken: {time.time() - start_time}")

output_file = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_{args.setting}.yaml")
with open(output_file, 'w') as f:
    yaml.dump(result, f)
