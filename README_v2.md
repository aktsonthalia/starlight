Source code for the paper "Do Deep Neural Network Solutions form a Star Domain?"

TODO: insert arxiv link

## Instructions

### Source code

1. Clone this repository:

    ```
    $ git clone https://github.com/aktsonthalia/starlight
    ```

1. Create a new conda environment and install the requirements:

    ```
    $ conda create -n starlight python=3.9
    $ conda activate starlight
    $ pip install -r requirements.txt
    ```

### WandB

1. Set up your [WandB](https://wandb.ai) account. Note down the `entity` and `project` values for use in the experiments. 


### Pretrained models

You can download pretrained models as zip files. Once they have been downloaded, extract them.

1. [CIFAR10-ResNet18](https://drive.google.com/file/d/1g-TxEGbORtHmxVEefoJtk2yxSf_mHL28/view?usp=drive_link)

### Configurations

1. You will find config files in `configs/`. Open the configuration file for the experiment you wish to run.
2. Change the wandb `entity` and `project` values.
3. Run the script `model_paths/create_model_paths.py`:
   ```
   $ cd model_paths/
   $ python create_model_paths.py -f <PATH_TO_EXTRACTED_MODELS>
   $ cd ..
   ```
   
4. Run the training script:
   
   ```
   $ python main.py <PATH_TO_CONFIG_FILE>
   ```

### Running Experiments

The experiments use slurm `sbatch` jobs. In order to run them conveniently using the GSheets tool, you need to use 

```
$ conda activate <PATH_TO_CONDA_ENV> && export PROJECT_ROOT_PROVIDED_FOR_STUNED=$(pwd) && python STAI-tuned/src/stuned/run_from_csv/__main__.py --conda_env <PATH_TO_CONDA_ENV> --csv_path <LINK_TO_GSHEET>::<NAME_OF_WORKSHEET> 
```

This script downloads the GSheet, submits a separate `sbatch` job for each row, and updates the GSheet with the WandB URL to the experiment run. 

### Plotting results

You can use the scripts in `postprocessing/`. Please first put the relevant WandB links inside `results/wandb_links` (follow the example yaml file).

```
$ python make_star_model_hypothesis_plots.py -c cifar10_resnet18
```



## Acknowledgements

1. Thanks to my colleague, [Alexander Rubinstein](https://github.com/alexanderRubinstein/) for the [STAI-tuned tool](https://github.com/AlexanderRubinstein/STAI-tuned) that made running the experiments so much easier.

1. Thanks to the [ML Cloud at Universität Tübingen](https://portal.mlcloud.uni-tuebingen.de/) for providing and maintaining the compute resources that made the experiments possible.

1. Thanks to [WandB](https://wandb.ai) for their [academic researchers plan](https://wandb.ai/site/research).

1. Thanks to the following open-source repositories, code from which was used in this one.
   1. [ResNet](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py)
   2. [DenseNet](https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py)
   3. [rebasin](https://pypi.org/project/rebasin/)
