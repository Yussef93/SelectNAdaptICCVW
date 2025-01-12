import argparse
import copy
import torch
import logging

import selfsupervision.SupContrast.networks.resnet_big
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default, get_cfg_ours
from dassl.engine import build_trainer

from train import *
from yacs.config import CfgNode as CN
import models.lccs_module
import models.resnet_lccs
import trainers.lccs
import gc
import os
def main(args):
    cfg = setup_cfg_ours(args)
    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    trainer = build_trainer(cfg)
    name = trainer.get_model_names()[0]
    # Dassl.pytorch/dassl/engine/trainer.py
    #before = copy.deepcopy(trainer._models[name].backbone)
    trainer.load_model_nostrict(args.model_dir, epoch=args.load_epoch)
    ss_checkpoint = torch.load(args.ss_dir)
    #target_ss_checkpoint= torch.load(args.target_dir)
    trainer.resnet = "resnet50"
    trainer.byol_model.load_state_dict(ss_checkpoint['state_dict'],strict=False)
    trainer.get_ksupport_loaders_ours_unsup()
   
    cfg['MODEL']['BACKBONE']['NAME'] = 'resnet50_lccs'
    cfg['DATALOADER']['K_TRANSFORMS'] = 1
    trainer_select = build_trainer(cfg)
    trainer_select.support_loader_train_transform = copy.deepcopy(trainer.support_loader_train_transform)
    trainer_select.support_loader_test_transform = copy.deepcopy(trainer.support_loader_test_transform)
    trainer_select.eval_loader = copy.deepcopy(trainer.eval_loader)
    #del trainer
    #gc.collect()
    trainer_select.load_model_nostrict(args.model_dir, epoch=args.load_epoch)
    try:
        trainer_select._models[name].backbone.load_state_dict(ss_checkpoint['backbone_dict'], strict=False)
    except Exception:
        trainer_select._models[name].backbone.load_state_dict(trainer.byol_model.online_encoder.net.state_dict(), strict=False)
    del trainer
    gc.collect()
    trainer_select.model.backbone.load_state_dict(ss_checkpoint['backbone_dict'], strict=False)
    trainer_select.initialization_stage()
    trainer_select.gradient_update_stage()
    trainer_select.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='', help='path to dataset')
    parser.add_argument(
        '--output-dir', type=str, default='', help='output directory'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default='',
        help='checkpoint directory (from which the training resumes)'
    )
    parser.add_argument(
        '--seed', #!
        type=int,
        default=-1,
        help='only positive value enables a fixed seed'
    )
    parser.add_argument(
        '--source-domains',
        type=str,
        nargs='+',
        help='source domains for DA/DG'
    )
    parser.add_argument(
        '--target-domains',
        type=str,
        nargs='+',
        help='target domains for DA/DG'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation methods'
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '--dataset-config-file',
        type=str,
        default='',
        help='path to config file for dataset setup'
    )
    parser.add_argument(
        '--trainer', type=str, default='', help='name of trainer'
    )
    parser.add_argument(
        '--backbone', type=str, default='', help='name of CNN backbone'
    )
    parser.add_argument('--head', type=str, default='', help='name of head')
    parser.add_argument(
        '--model-dir',
        type=str,
        default='',
        help='load model from this directory for eval-only mode'
    )
    parser.add_argument(
        '--load-epoch',
        type=int,
        default=50,
        help='load model weights at this epoch for evaluation'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='modify config options using the command-line'
    )
    args = parser.parse_args()

    args.root = './DATA/lccs/imcls/data'
    args.seed = 1

    
    source_target_pair = [['amazon','webcam']]
    for source_target in source_target_pair:
        start = 3
        end = 4
        for seed in range(start,end):
            args.seed=seed
            args.source_domains = [source_target[0]]
            args.target_domains = [source_target[1]]
            args.config_file = 'configs/trainers/dg/mixstyle/office31.yaml'
            args.dataset_config_file = 'configs/datasets/da/office31.yaml'
            args.trainer= 'LCCSk5n155'
            args.output_dir = 'output_results/ours/unsup'+args.trainer+'/output_source_models_resnet50_sgdsource/office31/Vanilla_resnet18_ndomain2_batch128/'+args.source_domains[0]+'/'+args.target_domains[0]+'/seed'+str(args.seed)
            args.model_dir = 'output_source_models/office31/Vanilla_resnet50_ndomain1_batch32/' + args.source_domains[0] + '/seed' + str(args.seed)
            args.ss_dir = 'output_results/SelfSupervision/output_source_models_byol_resnet50_rerun/office31/BYOL_resnet50_ndomain1/' + \
                          args.source_domains[0] + '/' + args.target_domains[0] + '/seed' + str( args.seed) + '/model/model.pth.tar-100'
            args.opt = ['MODEL.BACKBONE.NAME', 'resnet50', 'DATALOADER.TRAIN_X.SAMPLER', 'RandomSampler', 'DATALOADER.TRAIN_X.N_DOMAIN', '1', 'DATALOADER.TRAIN_X.BATCH_SIZE', '128', 'OPTIM.MAX_EPOCH', '150', 'TRAIN.CHECKPOINT_FREQ', '150']
            args.config_file = 'configs/trainers/dg/mixstyle/office31.yaml'
            args.dataset_config_file = 'configs/datasets/da/office31.yaml'
            args.transforms = None
            main(args)
