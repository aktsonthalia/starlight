Source code for the paper "Do Deep Neural Network Solutions form a Star Domain?"

TODO: insert arxiv link

## Instructions

Currently, the experiments can only be performed on a slurm cluster. 

### Source code

1. Clone this repository:

    ```
    $ git clone https://github.com/aktsonthalia/starlight
    ```
1. Add `STAI-tuned` as a submodule:

    ```
    $ git submodule add https://github.com/aktsonthalia/STAI-tuned
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

1. Create a [Google Service Account](https://support.google.com/a/answer/7378726?hl=en). Store its credentials in `~/config/gauth/credentials.json` on your slurm account.
2. Copy the [template](https://docs.google.com/spreadsheets/d/1yUZd8F9TncKoWxkTS_WtCtXiPOEwnjCssiJyOU2I9gc/edit#gid=727279666).
3. Share the copied GSheet with your newly created service account.

### Settings

1. Open `configs/default_config.yaml` and make the necessary changes. 
2. Open the GSheet with additional configurations and make the necessary changes there, too. Each row represents one experiment and correspondingly one `sbatch` job.
   1. Set `slurm:output`, `slurm:time`, `slurm:partition` and `slurm:error` to reflect the settings that you would use in a slurm job.
   2. Set `delta:exp_type` to either `train_anchor` for regular models and `train_star` for star models.
   3. Set `whether_to_run` to 1 for the rows / experiments that you wish to run. 

### Running Experiments

The experiments use slurm `sbatch` jobs. In order to run them conveniently using the GSheets tool, you need to use 

```
python STAI-tuned/src/stuned/run_from_csv/__main__.py --conda_env <PATH_TO_CONDA_ENV> --csv_path <LINK_TO_GSHEET>::<NAME_OF_WORKSHEET> 
```

## Acknowledgements

1. Thanks to my colleague, [Alexander Rubinstein](https://github.com/alexanderRubinstein/) for the [STAI-tuned tool](https://github.com/AlexanderRubinstein/STAI-tuned) that made running the experiments so much easier.

1. Thanks to the [ML Cloud at Universität Tübingen](https://portal.mlcloud.uni-tuebingen.de/) for providing and maintaining the compute resources that made the experiments possible.

1. Thanks to [WandB](https://wandb.ai) for their [academic researchers plan](https://wandb.ai/site/research).

1. Thanks to the following open-source repositories, code from which was used in this one.
   1. [ResNet](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py)
   2. [DenseNet](https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py)
   3. [rebasin](https://pypi.org/project/rebasin/)
