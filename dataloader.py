"""
@Author: Du Yunhao
@Filename: dataloader.py
@Contact: dyh_bupt@163.com
@Time: 2022/8/29 19:42
@Discription: Dataloader
"""
import json
import math
import torch
import random
import numpy as np
from PIL import Image
from os.path import join
from numpy.random import choice
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from utils import *
from sampler import RandomIdentitySampler, ConsistentModalitySampler, RandomCameraSampler


def get_dataloader(opt, mode, show=False):
    if mode in ('train', 'auxiliary'):
        sample_mode = opt.train_frame_sample
        transform = get_transform(opt, 'train')
    elif mode in ('query', 'gallery'):
        sample_mode = opt.test_frame_sample
        transform = get_transform(opt, 'test')
    else:
        raise RuntimeError(f'Wrong dataloader mode {mode}.')

    if opt.dataset == 'BUPTCampus':
        dataset = BUPTCampus_Dataset(
            data_root=opt.data_root,
            mode=mode,
            sample=sample_mode,
            seq_len=opt.sequence_length,
            transform=transform,
            random_flip=opt.random_flip,
            fake=opt.fake,
        )
    else:
        raise RuntimeError(f'Dataset {opt.dataset} is not supported for now.')

    if show:
        dataset.show_information()

    if mode == 'train':
        if opt.train_sampler is None:
            dataloader = DataLoader(
                dataset,
                batch_size=opt.train_bs,
                shuffle=True,
                drop_last=True,
                num_workers=opt.num_workers,
            )
        elif opt.train_sampler == 'RandomIdentitySampler':
            sampler = RandomIdentitySampler(
                dataset,
                np=opt.train_bs // (opt.train_sampler_nc * opt.train_sampler_nt),
                nc=opt.train_sampler_nc,
                nt=opt.train_sampler_nt,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=opt.train_bs,
                sampler=sampler,
                drop_last=True,
                num_workers=opt.num_workers,
            )
    elif mode == 'auxiliary':
        if opt.auxiliary_sampler is None:
            dataloader = DataLoader(
                dataset,
                batch_size=opt.train_bs,
                shuffle=True,
                drop_last=True,
                num_workers=opt.num_workers,
            )
        elif opt.auxiliary_sampler == 'RandomCameraSampler':
            sampler = RandomCameraSampler(
                dataset,
                np=opt.train_bs // (opt.auxiliary_sampler_nc * opt.auxiliary_sampler_nt),
                nc=opt.auxiliary_sampler_nc,
                nt=opt.auxiliary_sampler_nt,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=opt.train_bs,
                sampler=sampler,
                drop_last=False,
                num_workers=opt.num_workers,
            )
    else:
        if opt.test_sampler is None:
            dataloader = DataLoader(
                dataset,
                batch_size=opt.test_bs,
                shuffle=False,
                drop_last=False,
                num_workers=opt.num_workers,
            )
        elif opt.test_sampler == 'ConsistentModalitySampler':
            sampler = ConsistentModalitySampler(
                dataset,
                batch_size=opt.test_bs,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=opt.test_bs,
                sampler=sampler,
                drop_last=False,
                num_workers=opt.num_workers,
            )
    return dataloader, dataset.get_class_num()


class BUPTCampus_Dataset(Dataset):
    def __init__(self, data_root, mode, sample, seq_len, transform, random_flip=False, fake=False):
        """
        :param data_root:
        :param mode: 'train', 'query', 'gallery'
        :param sample: 'dense', 'uniform'
        :param seq_len:
        :param transform:
        """
        assert mode in ('train', 'query', 'gallery', 'auxiliary')
        self.mode = mode
        self.sample = sample
        self.seq_len = seq_len
        self.data_root = data_root
        self.transform = transform
        self.random_flip = random_flip
        self.fake = fake

        self.data_info = self.parse_data()
        self.data_paths = json.load(open(join(data_root, '../data_paths.json')))

        self.pid2label = {pid: label for label, pid in enumerate(self.pids)}

    def parse_data(self):
        if self.mode == 'train':
            data_info = self._parse_data('../train.txt')
        elif self.mode == 'query':
            data_info = self._parse_data('../query.txt')
        elif self.mode == 'gallery':
            data_info = self._parse_data('../gallery.txt')
        elif self.mode == 'auxiliary':
            data_info = self._parse_data('../train_auxiliary.txt')
        return data_info

    def _parse_data(self, path):
        data_info, pids = [], []
        path = join(self.data_root, path)
        with open(path) as f:
            for line in f.readlines():
                obj_id, modality, camera, tracklet_id = line.strip().split(' ')
                data_info.append((obj_id, modality, camera, tracklet_id))
                pids.append(obj_id)
        self.pids = sorted(set(pids))
        return data_info

    def get_class_num(self):
        return len(self.pids)

    def fast_iteration(self):
        iteration = [
            [self.pid2label[obj_id], MODALITY[modality], CAMERA[camera]]
            for (obj_id, modality, camera, tracklet_id) in self.data_info
        ]
        return iter(iteration)

    def __getitem__(self, index):
        obj_id, modality, camera, tracklet_id = self.data_info[index]

        if self.mode in ('train', 'auxiliary'):
            """
            Please note that every sample has two modalities while training,
            which means that the final batch size is equal to `2*opt.train_bs`
            """
            if modality == 'RGB/IR':
                data_paths_ir = self.data_paths[obj_id]['IR'][camera][tracklet_id]
                data_paths_rgb = self.data_paths[obj_id]['RGB'][camera][tracklet_id]
                if self.fake:
                    data_paths_rgb = [x.replace('/RGB/', '/FakeIR/') for x in data_paths_rgb]
                tra_len = len(data_paths_ir)
            else:
                raise RuntimeError('Only modality RGB/IR is supported for training.')

            if self.sample == 'random':
                replace = tra_len < self.seq_len
                frame_idx = sorted(choice(range(tra_len), size=self.seq_len, replace=replace))
            elif self.sample == 'restricted_random':
                frame_idx = list()
                if tra_len >= self.seq_len:
                    step = tra_len / self.seq_len
                    tra_idx = list(range(tra_len))
                else:
                    step = 1
                    tra_idx = [0] * (self.seq_len - tra_len) + list(range(tra_len))
                for i in range(self.seq_len):
                    idx = tra_idx[int(i*step): int((i+1)*step)]
                    frame_idx += random.sample(idx, 1)
            else:
                raise RuntimeError(f'Wrong sampling method {self.sample}.')

            images_ir = torch.stack(
                [self.transform(
                    Image.open(join(self.data_root, data_paths_ir[idx])).convert('RGB'))
                 for idx in frame_idx],
                dim=0
            )  # [T,C,H,W]
            images_rgb = torch.stack(
                [self.transform(
                    Image.open(join(self.data_root, data_paths_rgb[idx])).convert('RGB'))
                 for idx in frame_idx],
                dim=0
            )  # [T,C,H,W]

            # If set random_flip in self.transform instead,
            # frames within the same tracklet may have different directions
            if self.random_flip and torch.rand(1) < 0.5:
                images_ir = images_ir.flip(-1)
                images_rgb = images_rgb.flip(-1)

            label = self.pid2label[obj_id]

            return images_rgb, images_ir, label, CAMERA[camera]

        else:
            data_paths = self.data_paths[obj_id][modality][camera][tracklet_id]
            if self.fake:
                data_paths = [x.replace('/RGB/', '/FakeIR/') for x in data_paths]
            tra_len = len(data_paths)

            if self.sample == 'dense':
                '''
                Sample all frames for a tracklet.
                Only batch_size=1 is supported for this mode.
                '''
                frame_idx = range(tra_len)
            elif self.sample == 'uniform':
                '''
                Uniform sampling frames for a tracklet.
                '''
                frame_idx = np.linspace(0, tra_len, self.seq_len, endpoint=False, dtype=int)
            elif self.sample == 'first_half':
                frame_idx = np.linspace(0, tra_len//2, self.seq_len, endpoint=False, dtype=int)
            elif self.sample == 'second_half':
                frame_idx = np.linspace(tra_len//2, tra_len, self.seq_len, endpoint=False, dtype=int)
            else:
                raise RuntimeError(f'Wrong sampling method {self.sample}.')

            images = torch.stack(
                [self.transform(Image.open(join(self.data_root, data_paths[idx])).convert('RGB'))
                 for idx in frame_idx],
                dim=0
            )  # [T,C,H,W]

            return images, int(obj_id), CAMERA[camera], MODALITY[modality]

    def __len__(self):
        return len(self.data_info)

    def show_information(self):
        if self.mode in ('train', 'auxiliary'):
            factor = 2
        else:
            factor = 1
        print(
            f"===> MCPRL-ReID Dataset ({self.mode}) <===\n"
            f"Number of identities: {len(self.pids)}\n"
            f"Number of samples   : {len(self.data_info) * factor}"
        )
