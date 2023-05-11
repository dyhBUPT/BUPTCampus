"""
@Author: Du Yunhao
@Filename: test.py
@Contact: dyh_bupt@163.com
@Time: 2022/8/29 21:34
@Discription: test
"""
import os
import torch
from tqdm import tqdm
import torch.nn.functional as F

from model import get_model
from dataloader import get_dataloader

from opts import opt
from evaluation import evaluate, print_metrics
from utils import *


def test(model, dataloader_query, dataloader_gallery, show=False, save_dir='', return_all=False, postfix=''):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    print('========== Testing ==========')
    model.eval()
    with torch.no_grad():
        # query
        query_feats, query_pids, query_modals, query_cids = [], [], [], []
        for batch_idx, (imgs, pids, cids, modals) in enumerate(tqdm(dataloader_query)):
            imgs, cids = imgs.cuda(), cids.cuda()
            modal = modals[0]
            if modal == 0:
                feats = model(x_rgb=imgs)
            elif modal == 1:
                feats = model(x_ir=imgs)
            else:
                continue
            query_feats.append(feats)
            query_pids.append(pids)
            query_cids.append(cids)
            query_modals.append(modal.repeat(pids.size()))
        query_feats = torch.cat(query_feats, dim=0)  # [Nq, C]
        query_pids = torch.cat(query_pids, dim=0)  # [Nq,]
        query_modals = torch.cat(query_modals, dim=0)
        query_cids = torch.cat(query_cids, dim=0)

        # gallery
        gallery_feats, gallery_pids, gallery_modals, gallery_cids = [], [], [], []
        for batch_idx, (imgs, pids, cids, modals) in enumerate(tqdm(dataloader_gallery)):
            imgs, cids = imgs.cuda(), cids.cuda()
            modal = modals[0]
            assert modals.eq(modal).all()
            if modal == 0:
                feats = model(x_rgb=imgs)
            elif modal == 1:
                feats = model(x_ir=imgs)
            else:
                continue
            gallery_feats.append(feats)
            gallery_pids.append(pids)
            gallery_cids.append(cids)
            gallery_modals.append(modal.repeat(pids.size()))
        gallery_feats = torch.cat(gallery_feats, dim=0)  # [Ng, C]
        gallery_pids = torch.cat(gallery_pids, dim=0)  # [Ng,]
        gallery_modals = torch.cat(gallery_modals, dim=0)
        gallery_cids = torch.cat(gallery_cids, dim=0)

        # save
        if save_dir:
            torch.save(query_feats, join(save_dir, f'query_feats{postfix}.pth'))
            torch.save(query_pids, join(save_dir, 'query_pids.pth'))
            torch.save(query_modals, join(save_dir, 'query_modals.pth'))
            torch.save(query_cids, join(save_dir, 'query_cids.pth'))
            torch.save(gallery_feats, join(save_dir, f'gallery_feats{postfix}.pth'))
            torch.save(gallery_pids, join(save_dir, 'gallery_pids.pth'))
            torch.save(gallery_modals, join(save_dir, 'gallery_modals.pth'))
            torch.save(gallery_cids, join(save_dir, 'gallery_cids.pth'))

    # distance
    if opt.distance == 'cosine':
        distance = 1 - query_feats @ gallery_feats.T
    else:
        distance = euclidean_dist(query_feats, gallery_feats)

    CMC, MAP = [], []

    # evaluate (intra/inter-modality)
    for q_modal in (0, 1):
        for g_modal in (0, 1):
            q_mask = query_modals == q_modal
            g_mask = gallery_modals == g_modal
            tmp_distance = distance[q_mask, :][:, g_mask]
            tmp_qid = query_pids[q_mask]
            tmp_gid = gallery_pids[g_mask]
            tmp_cmc, tmp_ap = evaluate(tmp_distance, tmp_qid, tmp_gid, opt)
            CMC.append(tmp_cmc * 100)
            MAP.append(tmp_ap * 100)
            if show:
                print_metrics(
                    tmp_cmc, tmp_ap,
                    prefix='{:<3}->{:<3}:  '.format(MODALITY_[q_modal], MODALITY_[g_modal])
                )

    # evaluate (omni-modality)
    cmc, ap = evaluate(distance, query_pids, gallery_pids, opt)
    CMC.append(cmc * 100)
    MAP.append(ap * 100)

    if show:
        print_metrics(cmc, ap, prefix='AllModal:  ')

    del query_feats, query_pids, query_modals, gallery_feats, gallery_pids, gallery_modals, distance

    if return_all:
        return CMC, MAP
    else:
        return cmc * 100, ap * 100


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    model = get_model(opt, class_num=1074)
    model, _ = load_from_ckpt(model, 'model', opt.test_ckpt_path)

    postfix = ''
    if 'real' in opt.test_ckpt_path:
        postfix += '_real'
    elif 'fake' in opt.test_ckpt_path:
        postfix += '_fake'
    if 'auxiliary' in opt.test_ckpt_path:
        postfix += '-aux'

    frame_samples = opt.test_frame_sample.split('-')
    for frame_sample in frame_samples:
        print('Frame Sample: {}'.format(frame_sample))
        opt.test_frame_sample = frame_sample
        dataloader_query, _ = get_dataloader(opt, 'query', True)
        dataloader_gallery, _ = get_dataloader(opt, 'gallery', True)
        if frame_sample == 'uniform':
            curr_postfix = postfix + '_all'
        elif frame_sample == 'first_half':
            curr_postfix = postfix + '_first'
        elif frame_sample == 'second_half':
            curr_postfix = postfix + '_second'
        cmc, ap = test(
            model,
            dataloader_query, dataloader_gallery,
            show=True,
            postfix=curr_postfix,
            save_dir=opt.save_dir,
        )
