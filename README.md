# Test-Time Discovery via Hashing Memory

We conduct our work mainly based on the environment and framework of "PromptCCD: Learning Gaussian Mixture Prompt Pool for Continual Category Discovery"

## Environment

The environment can be easily installed through [conda](https://docs.conda.io/projects/miniconda/en/latest/) and pip. After cloning this repository, run the following command:
```shell
$ conda create -n ttd python=3.10
$ conda activate ttd

$ pip install scipy scikit-learn seaborn tensorboard kmeans-pytorch tensorboard opencv-python tqdm pycave timm
$ conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

*After setting up the environment, it’s recommended to restart the kernel.

## Data & Setup
Please refer to [README-data.md](doc/README-data.md) for more information regarding how to prepare and structure the datasets.


## Training
Please refer to [README-config.md](doc/README-config.md) for more information regarding the model configuration.

The configuration files for training and testing can be access at `config/%DATASET%/*.yaml`, organized based on different training datasets, and prompt module type.
For example, to train where C (*number of categories*) is known on CIFAR100, run:

```shell
$ CUDA_VISIBLE_DEVICES=%GPU_INDEX% python main.py
```
 and to change the config file or change the dataset, following code can be revised in `main.py`

```
parser.add_argument('--config', type=str, default="config/cifar100/cifar100_ttd_l2p2s.yaml", help='config file')
```

The training script will generate a directory in `exp/%SAVE_PATH%` where `%SAVE_PATH%` can be specified in the `"configs/%DATASET%/*.yaml"` file. 
 All of the necessary outputs, e.g., training ckpt, learned gmm for each stage, and experiment results are stored inside the directory. 

 The file structure should be:
```
ttd
├── config/
|   └── %DATASET%/
:       └── *.yaml (model configuration)
|
└── exp/
    └── %SAVE PATH%/
        ├── *.yaml (copied model configurations)
        ├── gmm/
        ├── model/ (training ckpt for each stage)
        ├── pred_labels/ (predicted labels from unlabelled images)
        ├── log_Kmeans_eval_stage_%STAGE%.txt
        └── log_SS-
```

