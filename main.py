import argparse

import torch
from util import config, util
from data import build_TTD_dataset, build_TTD_dataset_test
from tools import RunContinualTrainer

import warnings
warnings.filterwarnings("ignore")


def get_parser():
    '''
    Input: takes arguments from yaml config file located in config/$DATASET$/*.yaml
    Return: a dict with key for argument query
    '''
    parser = argparse.ArgumentParser(description='PyTorch implementation of ProTTD')
    parser.add_argument('--exp_name', type=str, default='ViT_ssk', help='To differentiate each experiments')

    parser.add_argument('--config', type=str, default="config/cifar100/cifar100_ttd_l2p2s.yaml", help='config file')
    # parser.add_argument('--config', type=str, default="config/tinyimagenet/tiny_ttd_l2p2s.yaml", help='config file')
    # parser.add_argument('--config', type=str, default="config/cub/cub200_ttd_l2p2s.yaml", help='config file')
    
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    # Check if config file is specified
    try: assert args.config is not None
    except: raise ValueError("Please specify config file")
    
    # Check if train or test mode is specified
    try: assert args.train != args.test
    except: raise ValueError("Please specify either train or test mode")
    
    # Load arguments from config file
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_cfg(cfg, vars(args))

    return cfg, args.config


def set_debug_apis(state: bool = False):
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state)
    torch.autograd.set_detect_anomaly(mode=state)


def main():
    args, config_path = get_parser()

    util.seed_everything(args.manual_seed)
    util.check_model_dir(args.save_path, config_path, args.train)

    util.info(f"Creating {args.dataset}'s dataset for TTD task ...")
    datasets_train = build_TTD_dataset(args=args, split='train')
    datasets_val = build_TTD_dataset(args=args, split='val')
    datasets_test = build_TTD_dataset_test(args=args, split='test')
    
    if args.run_ttd: 
        RunContinualTrainer(args, datasets_train, datasets_val, datasets_test)

    util.info("Program finished ...")


if __name__ == "__main__":
    set_debug_apis(state=False)
    main()