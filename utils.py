import copy
import dropbox
import numpy as np
import os
import random
import time
import torch
import torchmetrics
import torch.nn.functional as F
import unittest
import wandb

from torch import nn

from rebasin import PermutationCoordinateDescent

from models import models_dict

assertions = unittest.TestCase()

def check_config_for_training_experiment(config, config_path, logger):
    pass

def make_exps_deterministic(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # to suppress warning
    torch.use_deterministic_algorithms(True, warn_only=True)

def has_batch_norm(model):
    """Check if a model has batch norm layers.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            return True
    return False

def load_model_from_wandb_id(entity, project, wandb_id):

    for _ in range(10):
        try:
            api = wandb.Api()
            run = api.run(f"{entity}/{project}/{wandb_id}")
            checkpoints = [
                artifact
                for artifact in run.logged_artifacts()
                if artifact.type == "model-weights"
            ][0]
            num_epochs = run.config["training"]["num_epochs"]
            last_epoch_file = [
                f
                for f in checkpoints.files()
                if f.name == f"{wandb_id}_checkpoint{num_epochs-1}.pt"
            ][0]
            download_path = last_epoch_file.download(
                replace=True, root=os.environ["SCRATCH"]
            ).name
            state_dict = torch.load(download_path)
            return state_dict
        except:
            print("Failed to download model, trying again...")
            time.sleep(10)
            continue

@torch.no_grad()
def dataset_loss_and_accuracy(model, dl, loss_fn, model_list=None, ensemble=False, ensemble_type="average"):

    if ensemble:
        assert ensemble_type in ["average", "majority"], "ensemble_type must be one of ['average', 'majority']"
        print("evaluating ensemble")
        assert model_list is not None, "model_list must be provided if ensemble=True"
        assert model is None, "model must be None if ensemble=True"
        for model in model_list:
            model.eval()
            model.cuda()
    else:
        # print(f"evaluating single model, ensemble_type={ensemble_type}")
        assert model is not None, "model must be provided if ensemble=False"
        assert model_list is None, "model_list must be None if ensemble=False"
        model.eval()
        model.cuda()

    correct = 0
    total = 0
    loss_sum = 0

    for i, batch in enumerate(dl):
        x, y = batch
        x = x.cuda()
        y = y.cuda()

        if ensemble:
            # take mean of softmax outputs
            if ensemble_type == "average":
                # the "dim" argument:
                # collapse the specified dimension
                out = torch.mean(
                    torch.stack([F.softmax(model(x), dim=1) for model in model_list]), dim=0
                )
        else:
            out = model(x)
            loss = loss_fn(out, y)
            loss_sum += loss.detach().item()*len(y)

        correct += torch.sum(torch.argmax(out, dim=1) == y).item()
        total += len(y)

    final_loss = loss_sum / total
    final_accuracy = correct / total

    return final_loss if not ensemble else "N/A", final_accuracy


def setup_model(config):

    model_name = config.model.name
    model_settings = config.model.settings
    model = models_dict[model_name](**model_settings).cuda()

    if config.model.pretrained.use_pretrained:
        print("Loading pretrained model...")
        state_dict = load_model_from_wandb_id(
            config.logging.entity,
            config.logging.project,
            config.model.pretrained.wandb_id,
        )

        # for models trained with DataParallel, remove "module." from keys
        if all([key.startswith("module.") for key in state_dict.keys()]):
            state_dict = {key[7:]: value for key, value in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.cuda()

    if config.training.parallel:
        model = torch.nn.DataParallel(
            model, device_ids=list(range(torch.cuda.device_count()))
        )

    return model

def setup_optimizer(model, config):
    optimizer_name = config.training.optimizer.name
    assert optimizer_name in ["adam", "sgd", "adamw"]

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.optimizer.settings.learning_rate,
            weight_decay=config.training.optimizer.settings.weight_decay,
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.training.optimizer.settings.learning_rate,
            momentum=config.training.optimizer.settings.momentum,
            weight_decay=config.training.optimizer.settings.weight_decay,
            nesterov=config.training.optimizer.settings.nesterov,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.optimizer.settings.learning_rate,
            weight_decay=config.training.optimizer.settings.weight_decay,
        )

    return optimizer

class CosineAnnealingWithWarmup:
    """Cosine annealing with warmup
    If warmup_steps = 0, then this scheduler is equivalent to the cosine annealing scheduler.
    """
    def __init__(self, optimizer, config):

        self.optimizer = optimizer
        self.total_epochs = config.training.num_epochs
        self.current_epoch = -1

        self.warmup_epochs = config.training.scheduler.settings.warmup_steps
        # TODO: initial_lr is part of scheduler settings while learning_rate is part of optimizer settings
        # this is not ideal, change this sometime
        if self.warmup_epochs > 0:
            self.initial_lr = config.training.scheduler.settings.initial_lr
            self.peak_lr = config.training.optimizer.settings.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.initial_lr
        else: # no warmup
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = config.training.optimizer.settings.learning_rate
        
        self.cosine_annealing_epochs = self.total_epochs - self.warmup_epochs
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cosine_annealing_epochs
        )
    
    def step(self):

        self.current_epoch += 1
        if self.current_epoch < self.warmup_epochs: 
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.initial_lr + (self.peak_lr - self.initial_lr) * (self.current_epoch + 1) / self.warmup_epochs
        else:
            self.cosine_scheduler.step()

def setup_scheduler(optimizer, config):
    scheduler_name = config.training.scheduler.name
    assert scheduler_name in ["calr", "step", "no_sched", "cyclic"], f"scheduler_name must be one of ['calr', 'step', 'no_sched', 'cyclic'] but found {scheduler_name}"

    if scheduler_name == "calr":
        scheduler = CosineAnnealingWithWarmup(optimizer, config)

    if scheduler_name == "step":
        scheduler = StepLRWithMilestones(
            optimizer,
            total_epochs=config.training.num_epochs,
            gamma=config.training.scheduler.settings.step_lr_gamma, 
            milestones=config.training.scheduler.settings.step_lr_milestones
        )
    if scheduler_name == "no_sched":
        scheduler = None
    
    return scheduler

def interpolate_models(model1, model2, t, interpolated_model):

    model1_flattened = torch.cat([torch.reshape(param, (-1,)) for param in model1.parameters()])
    model2_flattened = torch.cat([torch.reshape(param, (-1,)) for param in model2.parameters()])
    interpolated_params = (1 - t) * model1_flattened + t * model2_flattened
    
    # Load interpolated parameters into interpolated_model
    index = 0
    for param in interpolated_model.parameters():
        param_shape = param.size()
        param_size = param.numel()
        param.data = interpolated_params[index : index + param_size].view(param_shape)
        index += param_size
    
    return interpolated_model

def match_weights(model1, model2, train_dl, recalculate_batch_statistics=False):
    """Perform weight matching between two models.
    Changes the weights of model2 to match the weights of model1.
    Copies the weights of model2 into a different model, so that the original model2 is not modified.
    """
    x = torch.randn((4, 3, train_dl.img_size, train_dl.img_size)).cuda()
    pcd = PermutationCoordinateDescent(
        model_a=model1, 
        model_b=model2,
        input_data_b=x, 
        device_a=torch.device("cuda:0"), 
        device_b=torch.device("cuda:0")
    )
    pcd.rebasin()
    del x
    # recalculate batch statistics if necessary
    if recalculate_batch_statistics and has_batch_norm(model2):
        model2.train()
        for i, batch in enumerate(train_dl):
            x, y = batch
            x = x.cuda()
            _ = model2(x)
            del x, _

    return model2

class DropboxSync:
    def __init__(self, access_token):
        self.dbx = dropbox.Dropbox(access_token)

    def upload_file(self, local_path, dropbox_path):
        with open(local_path, 'rb') as f:
            self.dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)

    def upload_folder(self, local_folder, dropbox_folder):
        for root, dirs, files in os.walk(local_folder):
            for filename in files:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_folder)
                dropbox_path = os.path.join(dropbox_folder, relative_path).replace(os.path.sep, '/')
                print(f'Uploading {local_path} to {dropbox_path}')
                self.upload_file(local_path, dropbox_path)

def load_models(config, base_model, mode="anchors"):

    models = []
    wandb_ids = None
    file_paths = None

    if mode == "anchors":
        try:
            wandb_ids = config.model.anchor_model_wandb_ids
            assert len(wandb_ids) > 0
        except:
            wandb_ids = None
            print(f"Loading anchor models from file paths... {config.model.anchor_model_paths}")
            with open(config.model.anchor_model_paths, "r") as f:
                file_paths = f.readlines()
        
    if mode == "held_out":
        try:
            wandb_ids = config.eval.held_out_anchors
            assert len(wandb_ids) > 0
        except:
            wandb_ids = None
            with open(config.eval.held_out_model_paths, "r") as f:
                file_paths = f.readlines()
    
    assert wandb_ids is not None or file_paths is not None, "wandb_ids and file_paths cannot both be None"

    if wandb_ids is not None:
        for wandb_id in wandb_ids:
            state_dict = load_model_from_wandb_id(
                config.logging.entity, 
                config.logging.project, 
                wandb_id
            )
            model = copy.deepcopy(base_model).cuda().eval()
            model.load_state_dict(state_dict)
            models.append(model)

    elif file_paths is not None:
        for file_path in file_paths:
            print(f"Loading model from file path: {file_path}")
            file_path = file_path.strip()
            state_dict = torch.load(file_path)['state_dict']
            # remove module
            if all([key.startswith("module.") for key in state_dict.keys()]):
                state_dict = {key[7:]: value for key, value in state_dict.items()}
            model = copy.deepcopy(base_model).cuda().eval()
            model.load_state_dict(state_dict)
            models.append(model)

    assert len(models) > 0, f"No models were loaded; wandb_ids={wandb_ids}, file_paths={file_paths}"
    return models

class StarDomain:
    def __init__(self, star_model, config, train_dl):

        self.star_model = star_model
        self.loss_sign = 1
        if config.exp_type in ["train_anti_star"]:
            self.loss_sign = -1
        self.train_dl = train_dl
        # load anchors
        self.anchor_models = load_models(config, star_model, mode="anchors")
        self.interpolated_model = copy.deepcopy(star_model).cuda()
        self.perform_battle_tests = config.perform_battle_tests

    def populate_star_model_gradients(self, batch, loss_fn, mu_star=0, mem_saving_mode=False):

        x, y = batch
        x = x.cuda()
        y = y.cuda()
        anchor_model = self.anchor_models[
            torch.randint(0, len(self.anchor_models), (1,)).item()
        ]
        anchor_model.cuda()
        t = torch.rand((1,)).item()
        self.interpolated_model = interpolate_models(
            self.star_model, 
            anchor_model, 
            t, 
            self.interpolated_model, 
        )

        # set gradients to None
        for param in self.interpolated_model.parameters():
            param.grad = None
            
        self.interpolated_model.train()
        
        # if batch norm, reset batch norm statistics
        for module in self.interpolated_model.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats()
            
        # check that the interpolated model's weights are the weighted average of the other two models' weights
        if self.perform_battle_tests:
            test_interpolation_was_carried_out(self.star_model, anchor_model, self.interpolated_model, t)

        out = self.interpolated_model(x)
        loss = loss_fn(out, y)
        if self.loss_sign == -1:
            # take reciprocal of loss
            loss = 1 / loss
            
        if mu_star > 0:
            out = self.star_model(x)
            loss_star = loss_fn(out, y)
            loss = (loss + mu_star * loss_star) / (1 + mu_star)

        loss.backward()

        for source_param, target_param in zip(
            self.interpolated_model.parameters(), self.star_model.parameters()
        ):
            if target_param.grad is None:
                target_param.grad = source_param.grad * (1 - t)
            else:
                target_param.grad += source_param.grad * (1 - t)
        
        if mem_saving_mode:
            anchor_model.cpu()
        return loss, out

    def recalculate_batch_statistics(self, dl):
        self.star_model.train()
        for i, batch in enumerate(dl):
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            out = self.star_model(x)
            del out
        self.star_model.eval()

    def align_anchors_with_star(self):
        print("Aligning anchors with star...")
        for i, anchor_model in enumerate(self.anchor_models):
            self.anchor_models[i] = match_weights(self.star_model, anchor_model, train_dl=self.train_dl)


def recalculate_batch_statistics(model, train_dl):
    model.cuda()
    model.train()
    for i, batch in enumerate(train_dl):
        x, y = batch
        x = x.cuda()
        y = y.cuda()
        _ = model(x)
    model.eval()
    return model

def average_model(model_list, train_dl=None):

    for model in model_list:
        model.eval()
    
    avg_state_dict = {}
    for key in model_list[0].state_dict().keys():
        avg_state_dict[key] = torch.mean(
            torch.stack([model.state_dict()[key].type(torch.FloatTensor) for model in model_list]), dim=0
        )
    
    avg_model = copy.deepcopy(model_list[0])
    avg_model.load_state_dict(avg_state_dict)

    if has_batch_norm(avg_model):
        if train_dl is None:
            raise Exception("train_dl must be provided if model has batch norm")
        recalculate_batch_statistics(avg_model, train_dl)

    return avg_model

def recalculate_batch_statistics(model, train_dl):
    model.cuda()
    model.train()
    for i, batch in enumerate(train_dl):
        x, y = batch
        x = x.cuda()
        y = y.cuda()
        _ = model(x)
    model.eval()
    return model

def make_interpolation_plot(model1, model2, dl, num_points, logger=None, plot_title="default title", loss_fn=F.cross_entropy, split="test", train_dl=None):

    bn = has_batch_norm(model1)

    loss_barrier = 0
    acc_barrier = 101

    model1.eval()
    model2.eval()

    model1_loss, model1_acc = dataset_loss_and_accuracy(model1, dl, loss_fn)
    model2_loss, model2_acc = dataset_loss_and_accuracy(model2, dl, loss_fn)
    
    ts = torch.linspace(0, 1, num_points)
    losses = []
    accuracies = []

    for t in ts:
        
        t = t.item()
        if t == 0:
            interpolated_model = copy.deepcopy(model1)
        elif t == 1:
            interpolated_model = copy.deepcopy(model2)
        else:
            interpolated_model = copy.deepcopy(model1)
            interpolated_model = interpolate_models(
                model1, model2, t, interpolated_model
            )
        interpolated_model.eval()

        if bn and t > 0 and t < 1:
            # recalculate batch statistics
            interpolated_model.train()
            for i, batch in enumerate(train_dl):
                x, y = batch
                x = x.cuda()
                y = y.cuda()
                _ = interpolated_model(x)
            interpolated_model.eval()

        loss, accuracy = dataset_loss_and_accuracy(
            interpolated_model, dl, loss_fn
        )

        loss_barrier_candidate = loss - ((1 - t) * model1_loss + t * model2_loss)
        acc_barrier_candidate = accuracy - ((1 - t) * model1_acc + t * model2_acc)

        if loss_barrier_candidate > loss_barrier:
            loss_barrier = loss_barrier_candidate
        if acc_barrier_candidate < acc_barrier:
            acc_barrier = acc_barrier_candidate

        losses.append(loss)
        accuracies.append(accuracy)
        # print(f"t: {t}, loss: {loss}, accuracy: {accuracy}")
    
    # plot in wandb
    data = [[t.item(), loss, acc] for t, loss, acc in zip(ts, losses, accuracies)]
    try:
        table = wandb.Table(data=data, columns=["t", f"{split}_loss", f"{split}_acc"])
        wandb.log(
            {
                f"{plot_title}_loss": wandb.plot.line(
                    table, "t", f"{split}_loss", title=f"{plot_title}_loss"
                )
            }
        )
        wandb.log(
            {
                f"{plot_title}_acc": wandb.plot.line(
                    table, "t", f"{split}_acc", title=f"{plot_title}_acc"
                )
            }
        )
    except:
        pass

    return loss_barrier, acc_barrier


class StepLRWithMilestones:

    def __init__(self, optimizer, total_epochs, gamma, milestones):

        self.FACTOR = gamma
        self.MILESTONES = milestones

        self.optimizer = optimizer
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_epoch = 0
        self.total_epochs = total_epochs
 
        # assert total_epochs in [1, 200, 400, 600], "total_epochs must be one of [1, 200, 400, 600]" 

    def step(self):

        self.current_epoch += 1
        if self.current_epoch in self.MILESTONES:        
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * self.FACTOR
        

# Battle tests

def test_only_last_layer_was_changed(model, config):
    try:
        pretrained_model_state_dict = load_model_from_wandb_id(
            config.logging.entity, config.logging.project, config.model.pretrained.wandb_id
        )
        current_state_dict = model.state_dict()
        # remove "module." from keys if model was trained with DataParallel
        if all([key.startswith("module.") for key in current_state_dict.keys()]):
            current_state_dict = {
                key[7:]: value
                for key, value in current_state_dict.items()
            }
        if all([key.startswith("module.") for key in pretrained_model_state_dict.keys()]):
            pretrained_model_state_dict = {
                key[7:]: value
                for key, value in pretrained_model_state_dict.items()
            }
        # verify that only the last layers differ
        for key in list(current_state_dict.keys())[:-2]:
            assert torch.all(
                current_state_dict[key] == pretrained_model_state_dict[key]
            )
    except:
        raise Exception("Failed to verify that only the last layers differ")

def test_interpolation_was_carried_out(model1, model2, interpolated_model, t):

    keys = list(model1.state_dict().keys())

    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()
    interpolated_model_state_dict = interpolated_model.state_dict()

    for key in keys:
        # first, check that the key exists in all three state dicts
        assert key in model1_state_dict.keys()
        assert key in model2_state_dict.keys()
        assert key in interpolated_model_state_dict.keys()
        
        # next, check that the interpolated model's weights are the weighted average of the other two models' weights
        for key in keys:
            assert torch.equal(
                interpolated_model_state_dict[key], 
                (1 - t) * model1_state_dict[key] + t * model2_state_dict[key]
            )

def test_average_of_three_identical_models_is_the_same_model(model, dl, loss_fn=F.cross_entropy):

    model1 = copy.deepcopy(model)
    model2 = copy.deepcopy(model)
    model3 = copy.deepcopy(model)
    model1.cuda()
    model2.cuda()
    model3.cuda()
    model1.eval()
    model2.eval()
    model3.eval()

    avg_model = average_model([model1, model2, model3])
    avg_model.eval()

    model1_loss, model1_acc = dataset_loss_and_accuracy(model1, dl, loss_fn)
    model2_loss, model2_acc = dataset_loss_and_accuracy(model2, dl, loss_fn)
    model3_loss, model3_acc = dataset_loss_and_accuracy(model3, dl, loss_fn)
    avg_model_loss, avg_model_acc = dataset_loss_and_accuracy(avg_model, dl, loss_fn)

    assertions.assertAlmostEqual(model1_loss, model2_loss)
    assertions.assertAlmostEqual(model1_loss, model3_loss)
    assertions.assertAlmostEqual(model1_loss, avg_model_loss)
    assertions.assertAlmostEqual(model1_acc, model2_acc)
    assertions.assertAlmostEqual(model1_acc, model3_acc)
    assertions.assertAlmostEqual(model1_acc, avg_model_acc)

    print(f"model1_loss: {model1_loss}")
    print(f"model2_loss: {model2_loss}")
    print(f"model3_loss: {model3_loss}")
    print(f"avg_model_loss: {avg_model_loss}")
    print(f"model1_acc: {model1_acc}")
    print(f"model2_acc: {model2_acc}")
    print(f"model3_acc: {model3_acc}")
    print(f"avg_model_acc: {avg_model_acc}")


def wandb_links_to_wandb_ids(filename="tmp.txt"):

    with open(filename, "r") as f:
        links = f.readlines()
    
    links = [link.strip() for link in links]
    links = [link.split("/")[-1] for link in links]

    with open("tmp2.txt", "w") as f:
        
        for i in range(len(links)):

            out_str = ""
            out_str += "["

            out_str += " ".join(links[:i+1])
            out_str += "]"
            f.write(out_str + "\n")


def compute_calibration_error(model, dl, num_classes):

    ce = torchmetrics.CalibrationError(num_classes=num_classes, task='multiclass')
    model.cuda()
    model.eval()

    for batch in dl:
        x, y = batch
        x = x.cuda()
        y = y.cuda()
        out = model(x)
        ce.update(out, y)
    
    return ce.compute()


def flatten_model(model):
    return torch.cat([torch.reshape(param, (-1,)) for param in model.parameters()])

def model_norm(model):
    return torch.norm(flatten_model(model), p=2)

def model_distance(model1, model2, train_dl=None, permute=False):

    if permute:
        assert train_dl is not None
        model2 = match_weights(
            model1, 
            copy.deepcopy(model2), 
            train_dl, 
            recalculate_batch_statistics=False
        )
    distance = torch.norm(flatten_model(model1) - flatten_model(model2), p=2)
    return distance.item()


@torch.no_grad()
def extensive_evaluation(model, dl):

    ensemble = isinstance(model, list)
    if not ensemble:
        model.cuda().eval()
    else:
        model = [m.cuda().eval() for m in model]

    correct = 0
    total = 0
    confidence_auroc_using_probs = torchmetrics.classification.BinaryAUROC(thresholds=20).cuda()
    confidence_auroc_using_entropy = torchmetrics.classification.BinaryAUROC(thresholds=20).cuda()
    ece = torchmetrics.CalibrationError(num_classes=dl.num_classes, task='multiclass').cpu()
    ece_misclassification = torchmetrics.CalibrationError(num_classes=2, task='binary').cpu()

    for x, y in dl:
        x = x.cuda()
        y = y.cuda()

        if ensemble:
            probs_list = []
            for m in model:
                probs_list.append(F.softmax(m(x), dim=1))
            probs = torch.mean(torch.stack(probs_list), dim=0)
            # probs = torch.mean(
                # torch.stack([F.softmax(m(x), dim=1) for m in model]), dim=0
            # )
        else:
            probs = F.softmax(model(x), dim=1)
        
        # probs = probs.cpu()
        # y = y.cpu()
        ece.update(probs, y)
        preds = torch.argmax(probs, dim=1)
        # preds = preds.cpu()
        correct += torch.sum(preds==y).item()
        total += len(y)

        max_probs, _ = torch.max(probs, dim=1)
        confidence_auroc_using_probs.update(max_probs, preds==y)
        entropy = -torch.sum(probs * torch.log(probs), dim=1)
        confidence_auroc_using_entropy.update(torch.ones_like(entropy) - entropy, preds==y)
        ece_misclassification.update(max_probs, preds==y)

    acc = (correct / total) * 100
    confidence_auroc_using_probs = confidence_auroc_using_probs.compute().item()
    confidence_auroc_using_entropy = confidence_auroc_using_entropy.compute().item()
    ece = ece.compute().item()
    ece_misclassification = ece_misclassification.compute().item()

    return {
        "acc": acc,
        "confidence_auroc_using_probs": confidence_auroc_using_probs,
        "confidence_auroc_using_entropy": confidence_auroc_using_entropy,
        "ece": ece,
        "ece_misclassification": ece_misclassification,
    }   