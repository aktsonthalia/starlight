import wandb
api = wandb.Api()
import os
import sys
sys.path.append("..")
from postprocessing.constants import *
from utils import DropboxSync
import json
from dotmap import DotMap
import argparse
from matplotlib.ticker import MaxNLocator

plt.rcParams['font.size'] = 14

with open("bma_results.json", "r") as f:
    results = json.load(f)

# plot
breakpoint()