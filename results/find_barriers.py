print("Importing libraries")
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

print("Imports successful")

ENTITY = "mode-connect"
PROJECT = "star-domain"
INTERPOLATION_NUM_POINTS = 11
HELD_OUT_NUM_SAMPLES = 5
ANCHOR_NUM_SAMPLES = 5
SEED = 42

SPECIAL_STAR_MODELS = {
    'resnet18_cifar10_sgd': 'by2vpp9d',
    'resnet18_cifar100_sgd': 'rylbd95p',
}

random.seed(SEED)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help="dataset")
parser.add_argument('--input_file', type=str, required=True, help="input file")
parser.add_argument('--model', type=str, required=True, help="model")
parser.add_argument('--output_dir', type=str, required=True, help="output file")
parser.add_argument('--setting', type=str, required=True, help="setting")
args = parser.parse_args()

config_name = f"{args.model}_{args.dataset}_{args.setting}"

# special cases go here
if config_name == 'resnet18_cifar10_warmup_mixed':
    HELD_OUT_NUM_SAMPLES = 6
    ANCHOR_NUM_SAMPLES = 10

if 'imagenet' in config_name:
    HELD_OUT_NUM_SAMPLES = 1
    ANCHOR_NUM_SAMPLES = 1

# end of special cases


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
    "anchor_held_out": [],
    "star_anchor": []
}

# get config
api = wandb.Api()
lmw = partial(load_model_from_wandb_id, entity=ENTITY, project=PROJECT)
run = api.run(f"{ENTITY}/{PROJECT}/{links_to_use['held_out'][0]}")
config = DotMap(run.config)

# setup dataset
train_dl_with_aug, val_dl, test_dl = datasets_dict[config.dataset.name](
    **config.dataset.settings
)

if 'imagenet' in config_name:
    train_dl_without_aug, _, _ = datasets_dict[config.dataset.name](
        batch_size=500,
        use_augmentation=False,
    )
else:
    train_dl_without_aug, _, _ = datasets_dict[config.dataset.name](
        batch_size=500,
    )

# setup model pairs

model_id_pairs = {
    "star_held_out": [],
    "star_anchor": [],
    "anchor_held_out": []
}

# reduce number of held-outs and anchors to SAMPLES

try:
    links_to_use['held_out'] = random.sample(links_to_use['held_out'], HELD_OUT_NUM_SAMPLES)
except: 
    pass

try:
    links_to_use['anchors'] = random.sample(links_to_use['anchors'], ANCHOR_NUM_SAMPLES)
except:
    pass


for held_out_link in links_to_use['held_out']:
    held_out_id = held_out_link.split('/')[-1]
    for star_link in links_to_use['stars']:
        # if config_name in SPECIAL_STAR_MODELS.keys():
        #     if star_link.split('/')[-1] != SPECIAL_STAR_MODELS[config_name]:
        #         continue
        star_id = star_link.split('/')[-1]
        model_id_pairs["star_held_out"].append((star_id, held_out_id))
    for anchor_link in links_to_use['anchors']:
        anchor_id = anchor_link.split('/')[-1]
        model_id_pairs["anchor_held_out"].append((anchor_id, held_out_id))

for anchor_link in links_to_use['anchors']:
    anchor_id = anchor_link.split('/')[-1]
    for star_link in links_to_use['stars']:
        if config_name in SPECIAL_STAR_MODELS.keys():
            if star_link.split('/')[-1] != SPECIAL_STAR_MODELS[config_name]:
                continue
        star_id = star_link.split('/')[-1]
        model_id_pairs["star_anchor"].append((star_id, anchor_id))

print(f"Number of star_held_out pairs: {len(model_id_pairs['star_held_out'])}")
print(f"Number of anchor_held_out pairs: {len(model_id_pairs['anchor_held_out'])}")
print(f"Number of star_anchor pairs: {len(model_id_pairs['star_anchor'])}")

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
            train_dl_with_aug, 
            recalculate_batch_statistics=True
        )

        print(f"Making interpolation plot between {model1_id} and {model2_id}")
        temp = make_interpolation_plot(
            model1=model1,
            model2=model2,
            num_points=INTERPOLATION_NUM_POINTS,
            logger=None,
            plot_title=f"Interpolation between {model1_id} and {model2_id}",
            train_dl=train_dl_without_aug,
            dl_to_calculate_batch_stats=train_dl_with_aug,
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
