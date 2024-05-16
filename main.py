import os
import sys
import time
import torch
import wandb
import yaml

from dotmap import DotMap
from statistics import mean 

try:
    sys.path.append("STAI-tuned/src")
    from stuned.utility.helpers_for_main import prepare_wrapper_for_experiment
    from stuned.utility.logger import try_to_log_in_csv_in_batch
except:
    pass 

from dataloaders import datasets_dict
from utils import (
    compute_calibration_error,
    check_config_for_training_experiment,
    dataset_loss_and_accuracy,
    has_batch_norm,
    make_exps_deterministic,
    make_interpolation_plot,
    load_models,
    match_weights,
    setup_model,
    setup_optimizer,
    setup_scheduler,
    StarDomain
)

TRAIN_EXP_TYPES = [
    "train_anchor", 
    "train_star",
]

def training_experiment(config, logger):

    # setup wandb
    try:
        wandb_run = logger.wandb_run
    except:
        wandb_args = {
            "tags": config.logging.tags,
            "config": config,
            "mode": "online",
        }

        if not "WANDB_DIR" in os.environ.keys():
            wandb_args["dir"] = "wandb"
            os.makedirs(wandb_args["dir"], exist_ok=True)
        if not isinstance(config.logging.entity, DotMap):
            wandb_args["entity"] = config.logging.entity
        if not isinstance(config.logging.project, DotMap):
            wandb_args["project"] = config.logging.project

        wandb_run = wandb.init(**wandb_args)

    print(f"wandb url: {wandb_run.get_url()}")
    
    wandb_run.log_code(".")
    checkpoints = wandb.Artifact(
        f"{config.dataset.name}-{config.model.name}-weights", type="model-weights"
    )

    # setup data and model
    train_dl, val_dl, test_dl = datasets_dict[config.dataset.name](**config.dataset.settings)
    model = setup_model(config)
    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
    loss_fn_eval = torch.nn.CrossEntropyLoss()

    if config.exp_type in ["train_star"]:
        star_domain = StarDomain(model, config, train_dl)

    step = 0
    
    val_loss, val_accuracy = dataset_loss_and_accuracy(
        model=model, dl=val_dl, loss_fn=loss_fn_eval
    )
    print(f"before training, val loss: {val_loss}, val accuracy: {val_accuracy}")
    
    for epoch in range(config.training.num_epochs):

        model.train()
        if config.exp_type in ["train_star"] and config.model.permute_anchors:
            star_domain.align_anchors_with_star()

        train_loss = 0
        train_accuracy = 0

        for i, batch in enumerate(train_dl):

            step += 1
            if config.dataset.name == "imagenet1k":
                if i % 100 == 0:
                    print(f"epoch {epoch} step {step}")
            optimizer.zero_grad(set_to_none=True)

            y_backup = batch[1].clone().cuda() # for calculating training accuracy
            # main training operations
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            if config.exp_type in ["train_star"]:
                loss, out = star_domain.populate_star_model_gradients(
                    batch, 
                    loss_fn, 
                    config.training.mu_star
                )
            else:
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()

            if config.training.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)

            optimizer.step()
            loss.detach()
            train_loss += loss.item()   
            accuracy_tmp = torch.sum(torch.argmax(out, axis=1) == y_backup) / len(y_backup)
            train_accuracy += accuracy_tmp.item()
            
            del x, y, out, batch

        if scheduler is not None: scheduler.step()

        # if training a star model and using batch norm, recalculate batch stats
        if config.exp_type in ["train_star"] and has_batch_norm(model):
            star_domain.recalculate_batch_statistics(train_dl)

        # log epoch results

        epoch_results = {}
        val_loss, val_accuracy = dataset_loss_and_accuracy(
            model=model, dl=val_dl, loss_fn=loss_fn_eval
        )
        
        epoch_results["val_accuracy"] = val_accuracy
        epoch_results["val_loss"] = val_loss
        epoch_results["train_loss"] = train_loss / len(train_dl)
        epoch_results["train_accuracy"] = train_accuracy / len(train_dl)
        epoch_results["lr"] = [param_group["lr"] for param_group in optimizer.param_groups][0]
        
        for _ in range(10):
            try:
                wandb_run.log(epoch_results)
                break
            except:
                print("wandb logging failed, trying again")
                time.sleep(5)

    # save. model weights
    try:
        ckpt_file = f"{os.environ['SCRATCH']}/{wandb_run.id}_checkpoint{epoch}.pt"
    except:
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_file = f"checkpoints/{wandb_run.id}_checkpoint{epoch}.pt"
        
    state_dict = model.state_dict()
    if config.training.parallel:
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    torch.save(state_dict, ckpt_file)
    checkpoints.add_file(ckpt_file)
    for _ in range(10):
        try:
            wandb_run.log_artifact(checkpoints)
            break
        except:
            print("wandb logging failed, trying again")
            time.sleep(5)

    # calculate test loss and accuracy
    test_loss, test_accuracy = dataset_loss_and_accuracy(
        model=model, dl=test_dl, loss_fn=loss_fn_eval
    )

    # log everything in csv
    csv_results = []
    csv_results.append(["val_accuracy", f"{val_accuracy:.5f}"])
    csv_results.append(["val_loss", f"{val_loss:.3f}"])
    csv_results.append(["train_loss", f"{train_loss / len(train_dl):.3f}"])
    csv_results.append(["train_accuracy", f"{train_accuracy / len(train_dl):.5f}"])
    csv_results.append(["test_accuracy", f"{test_accuracy:.5f}"])
    csv_results.append(["test_loss", f"{test_loss:.3f}"])

    try:
        try_to_log_in_csv_in_batch(logger, csv_results)
    except:
        pass


def generic_experiment(config, logger, processes_to_kill_before_exiting):

    assert torch.cuda.is_available()
    config = DotMap(config)
    make_exps_deterministic(config.params.random_seed)

    training_experiment(config, logger)


if __name__ == "__main__":

    try:
        prepare_wrapper_for_experiment(check_config_for_training_experiment)(generic_experiment)()
    except:
        config_file = sys.argv[1]
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        generic_experiment(config, None, [])