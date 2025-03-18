# Test-Time Discovery via Hashing Memory ([Arxiv](https://arxiv.org/abs/2503.10699))

We introduce Test-Time Discovery (TTD) as a novel task that addresses class shifts during testing, requiring models to simultaneously identify emerging categories while preserving previously learned ones. A key challenge in TTD is distinguishing newly discovered classes from those already identified. To address this, we propose a training-free, hash-based memory mechanism that enhances class discovery through fine-grained comparisons with past test samples. Leveraging the characteristics of unknown classes, our approach introduces hash representation based on feature scale and directions, utilizing Locality-Sensitive Hashing (LSH) for efficient grouping of similar samples. This enables test samples to be easily and quickly compared with relevant past instances. Furthermore, we design a collaborative classification strategy, combining a prototype classifier for known classes with an LSH-based classifier for novel ones. To enhance reliability, we incorporate a self-correction mechanism that refines memory labels through hash-based neighbor retrieval, ensuring more stable and accurate class assignments. Experimental results demonstrate that our method achieves good discovery of novel categories while maintaining performance on known classes, establishing a new paradigm in model testing. 




## Environment

We conduct our work mainly based on the environment and framework of "PromptCCD: Learning Gaussian Mixture Prompt Pool for Continual Category Discovery ([Arxiv](https://arxiv.org/pdf/2407.19001))"

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

## Cite Us

```
@article{lyu2025testtimediscoveryhashingmemory,
      title={Test-Time Discovery via Hashing Memory}, 
      author={Fan Lyu and Tianle Liu and Zhang Zhang and Fuyuan Hu and Liang Wang},
      year={2025},
      journal={arXiv preprint arXiv:2503.10699},
      url={https://arxiv.org/abs/2503.10699}, 
}
```
