from constants import PROJECT_PATH
import sys
sys.path.append(PROJECT_PATH)   

import argparse
import yaml
import torch
import wandb 
from dotmap import DotMap
from utils import setup_model, make_interpolation_plot
from dataloaders import datasets_dict

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', '-c', type=str)
parser.add_argument('--model_a', '-a', type=str)
parser.add_argument('--model_b', '-b', type=str)
args = parser.parse_args()



config_file = args.config_file
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

config = DotMap(config)

wandb_run = wandb.init(
    entity=config.logging.entity,
    project=config.logging.project,
    job_type="plot_interpolation",
    config=config,
    reinit=True,
)

model1 = setup_model(config)
model2 = setup_model(config)

model1.load_state_dict(torch.load(args.model_a)["state_dict"])
model2.load_state_dict(torch.load(args.model_b)["state_dict"])

train_dl, val_dl, test_dl = datasets_dict[config.dataset.name](
    **config.dataset.settings
)

make_interpolation_plot(
    model1=model1,
    model2=model2,
    num_points=config.interpolation.num_points,
    logger=None,
    plot_title=f"Interpolation between {args.model_a.split('/')[-1]} and {args.model_b.split('/')[-1]}",
    train_dl=train_dl,
    dl=test_dl,
)