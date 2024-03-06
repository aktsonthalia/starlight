Source code for the paper "Do Deep Neural Network Solutions form a Star Domain?"

TODO: insert arxiv link

## Instructions

You will need to use a Slurm-based computing cluster to run the experiments.

### Source code

1. Clone this repository:

    ```
    $ git clone https://github.com/aktsonthalia/starlight
    ```
1. Add submodules:
    ```
    $ git submodule update --init
    $ git submodule update --recursive --remote
    ```
1. Create a new conda environment and install the requirements:

    ```
    $ conda create -n starlight python=3.9
    $ conda activate starlight
    $ pip install -r requirements.txt
    ```

### WandB

1. Set up your [WandB](https://wandb.ai) account. Note down the `entity` and `project` values for use in the experiments. 

### GSheets

We use Google Sheets to configure the experiments, and store results in a handy format.

1. Create a [Google Service Account](https://support.google.com/a/answer/7378726?hl=en). Store its credentials in `~/config/gauth/credentials.json` on your slurm account. Note that `~` refers to the value stored in your `$HOME` variable. Service accounts are an elegant way of accessing Google content remotely without manual authentication.
2. Copy the [template](https://docs.google.com/spreadsheets/d/1yUZd8F9TncKoWxkTS_WtCtXiPOEwnjCssiJyOU2I9gc/edit#gid=727279666) to your own Google account so that you can make changes to it.
3. Share the copied GSheet with your newly created service account, so that you can access the GSheet remotely.

### Settings

1. Open `configs/default_config.yaml` and make the necessary changes. 
2. Open the GSheet with additional configurations and make the necessary changes there, too. Each row represents one experiment and correspondingly one `sbatch` job.
   1. Set `slurm:output`, `slurm:time`, `slurm:partition` and `slurm:error` to reflect the settings that you would use in a slurm job.
   2. Set `delta:exp_type` to either `train_anchor` for regular models and `train_star` for star models.
   3. Set `whether_to_run` to 1 for the rows / experiments that you wish to run. 
   4. Set `delta:eval.held_out_anchors` and `delta:model.anchor_model_wandb_ids` to lists of WandB run IDs, after you have generated the requisite runs.

### Running Experiments

The experiments use slurm `sbatch` jobs. In order to run them conveniently using the GSheets tool, you need to use 

```
$ conda activate <PATH_TO_CONDA_ENV> && python STAI-tuned/src/stuned/run_from_csv/__main__.py --conda_env <PATH_TO_CONDA_ENV> --csv_path <LINK_TO_GSHEET>::<NAME_OF_WORKSHEET> 
```

This script downloads the GSheet, submits a separate `sbatch` job for each row, and updates the GSheet with the WandB URL to the experiment run. 

## Acknowledgements

1. Thanks to my colleague, [Alexander Rubinstein](https://github.com/alexanderRubinstein/) for the [STAI-tuned tool](https://github.com/AlexanderRubinstein/STAI-tuned) that made running the experiments so much easier.

1. Thanks to the [ML Cloud at Universität Tübingen](https://portal.mlcloud.uni-tuebingen.de/) for providing and maintaining the compute resources that made the experiments possible.

1. Thanks to [WandB](https://wandb.ai) for their [academic researchers plan](https://wandb.ai/site/research).

1. Thanks to the following open-source repositories, code from which was used in this one.
   1. [ResNet](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py)
   2. [DenseNet](https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py)
   3. [rebasin](https://pypi.org/project/rebasin/)
