"""
@Author: Du Yunhao
@Filename: evaluation.py
@Contact: dyh_bupt@163.com
@Time: 2022/8/30 16:39
@Discription: evaluation
"""
import torch
import numpy as np


def print_metrics(cmc, ap, prefix=''):
    print(
        '{}mAP: {:.2%} | Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%}.'
            .format(prefix, ap, cmc[0], cmc[4], cmc[9], cmc[19])
    )


def evaluate(distmat, query_pids, gallery_pids, opt):
    if isinstance(distmat, torch.Tensor):
        distmat = distmat.detach().cpu().numpy()
    if isinstance(query_pids, torch.Tensor):
        query_pids = query_pids.detach().cpu().numpy()
    if isinstance(gallery_pids, torch.Tensor):
        gallery_pids = gallery_pids.detach().cpu().numpy()

    num_q, num_g = distmat.shape
    assert num_q == len(query_pids) and num_g == len(gallery_pids)

    max_rank = min(opt.max_rank, num_g)

    indices = np.argsort(distmat, axis=1)
    matches = (gallery_pids[indices] == query_pids[:, np.newaxis]).astype(np.int32)

    num_valid_query = 0
    all_cmc, all_ap = [], []
    for qi in range(num_q):
        orig_cmc = matches[qi]

        # This condition is true when the query doesn't appear in gallery.
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_query += 1.

        # compute average precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc =  np.asarray(tmp_cmc) * orig_cmc
        ap = tmp_cmc.sum() / num_rel
        all_ap.append(ap)

    assert num_valid_query > 0, "No query appears in gallery."

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_query
    mAP = np.mean(all_ap)

    return all_cmc, mAP

