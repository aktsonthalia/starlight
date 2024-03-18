Source code for the paper [Do Deep Neural Network Solutions form a Star Domain?](https://arxiv.org/abs/2403.07968)

If you find this work useful, please consider citing:

```
@misc{sonthalia2024deep,
      title={Do Deep Neural Network Solutions Form a Star Domain?}, 
      author={Ankit Sonthalia and Alexander Rubinstein and Ehsan Abbasnejad and Seong Joon Oh},
      year={2024},
      eprint={2403.07968},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Instructions

### Source code :computer:

1. Clone this repository:

    ```
    $ git clone https://github.com/aktsonthalia/starlight --recursive
    ```

2. Create a new conda environment and install the requirements:

    ```
    $ conda create -n starlight python=3.9
    $ conda activate starlight
    $ pip install -r requirements.txt
    ```

### WandB :bar_chart:

3. Set up your [WandB](https://wandb.ai) account. Note down the `entity` and `project` values for use in the experiments. 


### Pretrained models :brain:

4. You can download pretrained models as zip files. Once they have been downloaded, extract them.

- [CIFAR10-ResNet18](https://drive.google.com/file/d/1g-TxEGbORtHmxVEefoJtk2yxSf_mHL28/view?usp=drive_link)
- [CIFAR100-ResNet18](https://drive.google.com/file/d/1zTH5_YBM9h4AYfeQWhkJyqXW_14xS38B/view?usp=drive_link)
- [CIFAR10-DenseNet](https://drive.google.com/file/d/1zTH5_YBM9h4AYfeQWhkJyqXW_14xS38B/view?usp=drive_link)
- [CIFAR100-DenseNet](https://drive.google.com/file/d/1ehiFGtrF-p3MKvHRBFDUA5qzyI0s1GiS/view?usp=sharing)
- [ImageNet1k-ResNet18](https://drive.google.com/file/d/1ehiFGtrF-p3MKvHRBFDUA5qzyI0s1GiS/view?usp=sharing)

5. Run the script `model_paths/create_model_paths.py`:
   ```
   $ cd model_paths/
   $ python create_model_paths.py -f <PATH_TO_EXTRACTED_MODELS>
   $ cd ..
   ```

### Configuration :gear:

6. You will find config files in `configs/`. Open the configuration file for the experiment you wish to run (names are self-explanatory).
   - `cifar10_resnet18_star.yaml`
7. Change the wandb `entity` and `project` values.
8.  Go to `dataloaders/constants.py` and change any dataset paths that you might need to. Otherwise, default values will be used.

### Running experiments :test_tube:

9.  Run the training script:
   
   ```
   $ python main.py <PATH_TO_CONFIG_FILE>
   ```

### Plotting :chart_with_upwards_trend::chart_with_downwards_trend:

You can plot the loss barrier between two given models using:

```
$ cd postprocessing
$ python plot_barriers.py -c <CONFIG_FILE from configs/> -a <PATH_TO_MODEL_A_CHECKPOINT> -b <PATH_TO_MODEL_B_CHECKPOINT>
```

## Acknowledgements :clap:

1. Thanks to my collaborator, [Alexander Rubinstein](https://github.com/alexanderRubinstein/) for the [STAI-tuned tool](https://github.com/AlexanderRubinstein/STAI-tuned) that made running the experiments so much easier. If you're interested, you can run more extensive experiments using this tool, too. Follow [this README](README_stuned.md).
   
2. Thanks to [Arnas Uselis](https://github.com/oshapio/) for testing the code prior to release.

3. Thanks to the [ML Cloud at Universität Tübingen](https://portal.mlcloud.uni-tuebingen.de/) for providing and maintaining the compute resources that made the experiments possible.

4. Thanks to [WandB](https://wandb.ai) for their [academic researchers plan](https://wandb.ai/site/research).

5. Thanks to the following open-source repositories, code from which was used in this one.
   1. [ResNet](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py).
   2. [DenseNet](https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py).
   3. [rebasin](https://pypi.org/project/rebasin/).
   4. [ffcv](https://github.com/libffcv/ffcv)

## Issues and Updates 🔧

Feel free to open an issue, if you find a bug, or would like to see a new feature.

Check [the issues](https://github.com/aktsonthalia/starlight/labels/enhancement) for planned improvements. If you wish, you may also open pull requests addressing them.
