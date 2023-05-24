"""
@Author: Du Yunhao
@Filename: opts.py
@Contact: dyh_bupt@163.com
@Time: 2022/8/29 19:35
@Discription: options
"""
import json
import argparse
from os.path import join


class opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # basic settings
        self.parser.add_argument('--gpus', type=str, default='0')
        self.parser.add_argument('--dataset', type=str, default='BUPTCampus')
        self.parser.add_argument('--gpu_mode', type=str, default='dp', help='single/dp/ddp')
        self.parser.add_argument('--data_root', type=str, default='/data1/dyh/data/VIData/BUPTCampus/DATA')
        self.parser.add_argument('--save_dir', type=str, default='/data1/dyh/results/BUPTCampus/tmp')
        self.parser.add_argument('--fake', action='store_true', default=False)
        self.parser.add_argument('--feature_postfix', type=str, default='')

        # basic parameters
        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--sequence_length', type=int, default=10)
        self.parser.add_argument('--img_hw', nargs='+', type=int, default=(256, 128))
        self.parser.add_argument('--norm_std', type=list, default=[0.229, 0.224, 0.225])
        self.parser.add_argument('--norm_mean', type=list, default=[0.485, 0.456, 0.406])

        # model
        self.parser.add_argument('--temporal', type=str, default='gap', help='gap/self-attention')
        self.parser.add_argument('--backbone', type=str, default='resnet34')
        self.parser.add_argument('--one_stream', action='store_true', default=False)

        # training
        self.parser.add_argument('--train_bs', type=int, default=16)
        self.parser.add_argument('--base_lr', type=float, default=2e-4)
        self.parser.add_argument('--max_epoch', type=int, default=100)
        self.parser.add_argument('--padding', type=int, default=10)
        self.parser.add_argument('--eval_freq', type=int, default=1)
        self.parser.add_argument('--warmup_epoch', type=int, default=0)
        self.parser.add_argument('--warmup_start_lr', type=float, default=1e-5)
        self.parser.add_argument('--optimizer', type=str, default='Adam')
        self.parser.add_argument('--cosine_end_lr', type=float, default=0.)
        self.parser.add_argument('--weight_decay', type=float, default=1e-5)
        self.parser.add_argument('--train_print_freq', type=int, default=100)
        self.parser.add_argument('--triplet_margin', type=float, default=0.6)
        self.parser.add_argument('--triplet_hard', action='store_false', default=True)
        self.parser.add_argument('--train_frame_sample', type=str, default='random')
        self.parser.add_argument('--random_flip', action='store_false', default=True)
        self.parser.add_argument('--lambda_ce', type=float, default=1)
        self.parser.add_argument('--lambda_tri', type=float, default=1)

        # sampler
        self.parser.add_argument('--train_sampler', type=str,
                                 default='RandomIdentitySampler', help='None for shuffle')
        self.parser.add_argument('--train_sampler_nc', type=int, default=2)
        self.parser.add_argument('--train_sampler_nt', type=int, default=1)
        self.parser.add_argument('--auxiliary_sampler', type=str, default='RandomCameraSampler',
                                 help='None for shuffle, or RandomIdentitySampler/RandomCameraSampler')
        self.parser.add_argument('--auxiliary_sampler_nc', type=int, default=2)
        self.parser.add_argument('--auxiliary_sampler_nt', type=int, default=1)  # 1 or 2
        self.parser.add_argument('--test_sampler', type=str,
                                 default='ConsistentModalitySampler', help='None for no shuffle')

        # Auxiliary
        self.parser.add_argument('--auxiliary', action='store_true', default=False)
        self.parser.add_argument('--aux_phi', type=float, default=3)

        # resume
        self.parser.add_argument('--resume_path', type=str, default='')

        # testing
        self.parser.add_argument('--test_bs', type=int, default=64)  # Please don't change it.
        self.parser.add_argument('--test_frame_sample', type=str, default='uniform')
        self.parser.add_argument('--test_ckpt_path', type=str)
        self.parser.add_argument('--test_feat_path', type=str)

        # evaluation
        self.parser.add_argument('--max_rank', type=int, default=20)
        self.parser.add_argument('--distance', type=str, default='euclidean')
        self.parser.add_argument('--rerank_lambda', type=float, default=0.7)
        self.parser.add_argument('--rerank_k1', type=int, default=3)
        self.parser.add_argument('--rerank_k2', type=int, default=1)

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        
        return opt


opt = opts().parse()
