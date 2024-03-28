from models import models_dict
from dataloaders import datasets_dict
import wandb
from utils import (
    load_model_from_wandb_id,
    make_interpolation_plot,
    match_weights
)

ENTITY = "mode-connect"
PROJECT = "star-domain"
# load two vgg models

MODEL_SETTINGS = {
    "name": "vgg11",
    "settings": {
        "num_classes": 10
    }
}
DATASET_SETTINGS = {
    "name": "cifar10",
    "settings": {
        "batch_size": 128,
        "num_workers": 4
    }
}
WANDB_ID_1 = "jr8gqoit"
WANDB_ID_2 = "sukzyv2p"

train_dl, val_dl, test_dl = datasets_dict[DATASET_SETTINGS["name"]](**DATASET_SETTINGS["settings"])

model1 = models_dict[MODEL_SETTINGS["name"]](**MODEL_SETTINGS["settings"])
model1.load_state_dict(
    load_model_from_wandb_id(
        ENTITY, PROJECT, WANDB_ID_1
    )
)

model2 = models_dict[MODEL_SETTINGS["name"]](**MODEL_SETTINGS["settings"])
model2.load_state_dict(
    load_model_from_wandb_id(
        ENTITY, PROJECT, WANDB_ID_2
    )
)

# make wandb run

wandb_run = wandb.init(
    entity=ENTITY,
    project=PROJECT,
    job_type="interpolation",
    config={
        "model_settings": MODEL_SETTINGS,
        "dataset_settings": DATASET_SETTINGS,
        "wandb_id_1": WANDB_ID_1,
        "wandb_id_2": WANDB_ID_2
    }
)

make_interpolation_plot(
    model1=model1,
    model2=model2,
    train_dl=train_dl,
    dl=test_dl,
    plot_title=f"before matching"
)

