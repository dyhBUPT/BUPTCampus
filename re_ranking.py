"""
@Author: Du Yunhao
@Filename: re_ranking.py
@Contact: dyh_bupt@163.com
@Time: 2023/4/5 10:15
@Discription: k-reciprocal  re-ranking
"""
import os
import torch
import numpy as np
from time import time

from opts import opt
from evaluation import evaluate, print_metrics
from utils import *


class K_Reciprocal:
    """reference: https://github.com/michuanhaohao/reid-strong-baseline/blob/master/utils/re_ranking.py"""
    def __init__(self, k1, k2, lambda_value, alpha=1/2, beta=2/3, distance='euclidean'):
        self.k1 = k1
        self.k2 = k2
        self.lambda_value = lambda_value
        self.alpha = alpha
        self.beta = beta
        self.distance = distance

    def get_original_distance(self, x, y, norm=False):
        assert self.distance in ('cosine', 'euclidean')
        fn_norm = lambda x: x / np.sqrt(np.sum(x ** 2, axis=1))[:, np.newaxis]
        if norm:
            x, y = fn_norm(x), fn_norm(y)
        if self.distance == 'cosine':
            return 1 - np.dot(x, y.T)
        elif self.distance == 'euclidean':
            return np.sqrt(
                np.sum(x ** 2, axis=1)[:, np.newaxis] +
                np.sum(y ** 2, axis=1)[np.newaxis, :] -
                2 * np.dot(x, y.T) + 1e-5
            )

    def get_jaccard_distance(self, q_num, g_num, features, fast_version=True):
        """
        - fast_version: fast calculation based on some tricks. It runs much faster, but harder to read.
        """
        jaccard_dist = np.ones((q_num, g_num), dtype=np.float16)
        if fast_version:
            q_non_zero_index = [np.where(features[i, :] != 0)[0] for i in range(q_num)]
            g_non_zero_index = [np.where(features[:, j] != 0)[0] for j in range(q_num + g_num)]
            for i, query_feature in enumerate(features[:q_num]):
                minimum = np.zeros(q_num + g_num, dtype=np.float16)
                q_non_zero_index_i = q_non_zero_index[i]
                indices = [g_non_zero_index[idx] for idx in q_non_zero_index_i]
                for j in range(len(q_non_zero_index_i)):
                    minimum[indices[j]] += np.minimum(
                        features[i, q_non_zero_index_i[j]], features[indices[j], q_non_zero_index_i[j]])
                minimum = minimum[q_num:]
                jaccard_dist[i] = 1 - minimum / (2 - minimum)
        else:
            for i, query_feature in enumerate(features[:q_num]):
                for j, gallery_feature in enumerate(features[q_num:]):
                    minimum = np.minimum(query_feature, gallery_feature).sum()
                    maximum = np.maximum(query_feature, gallery_feature).sum()
                    jaccard_dist[i, j] = 1 - minimum / maximum
        return jaccard_dist

    def get_k_reciprocal_index(self, query_index, ranking_list, k):
        forward_k_neighbor_index = ranking_list[query_index, :k + 1]  # forward retrieval
        backward_k_neighbor_index = ranking_list[forward_k_neighbor_index, :k + 1]  # backward retrieval
        k_reciprocal_row = np.where(backward_k_neighbor_index == query_index)[0]
        k_reciprocal_index = forward_k_neighbor_index[k_reciprocal_row]
        return k_reciprocal_index

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()

    def __call__(self, query_feats, gallery_feats):
        """
        Call k-reciprocal re-ranking
        query_feats: [M,L]
        gallery_feats: [N,L]
        """
        '''1) original distance'''
        q_feats, g_feats = self._to_numpy(query_feats), self._to_numpy(gallery_feats)
        all_feats = np.concatenate((q_feats, g_feats), axis=0)
        q_num, g_num, all_num = q_feats.shape[0], g_feats.shape[0], all_feats.shape[0]
        original_dist = self.get_original_distance(all_feats, all_feats)  # [M+N, M+N]
        original_dist /= original_dist.max(axis=1)[:, np.newaxis]  # row normalization
        original_rank = np.argsort(original_dist).astype(int)  # original ranking list
        '''2) k-reciprocal features'''
        k_reciprocal_features = np.zeros_like(original_dist, dtype=np.float16)  # i.e., `V` in paper
        for i in range(all_num):
            k_reciprocal_index = self.get_k_reciprocal_index(i, original_rank, k=self.k1)
            k_reciprocal_incremental_index = k_reciprocal_index.copy()  # index after incrementally adding
            '''incrementally adding'''
            for j, candidate in enumerate(k_reciprocal_index):
                candidate_k_reciprocal_index = self.get_k_reciprocal_index(
                    candidate, original_rank, k=int(round(self.k1 * self.alpha)))
                if len(np.intersect1d(k_reciprocal_index, candidate_k_reciprocal_index)) \
                        > self.beta * len(candidate_k_reciprocal_index):
                    k_reciprocal_incremental_index = np.append(
                        k_reciprocal_incremental_index, candidate_k_reciprocal_index)
            k_reciprocal_incremental_index = np.unique(k_reciprocal_incremental_index)
            '''compute '''
            weight = np.exp(-original_dist[i, k_reciprocal_incremental_index])  # reassign weights with Gaussian kernel
            k_reciprocal_features[i, k_reciprocal_incremental_index] = weight / weight.sum()
        '''3) local query expansion'''
        if self.k2 != 1:
            k_reciprocal_expansion_features = np.zeros_like(k_reciprocal_features)
            for i in range(all_num):
                k_reciprocal_expansion_features[i, :] = \
                    np.mean(k_reciprocal_features[original_rank[i, :self.k2], :], axis=0)
            k_reciprocal_features = k_reciprocal_expansion_features
        '''4) Jaccard distance'''
        jaccard_dist = self.get_jaccard_distance(q_num, g_num, k_reciprocal_features, fast_version=True)
        return self.lambda_value * original_dist[:q_num, q_num:] + \
               (1 - self.lambda_value) * jaccard_dist


def get_features(mode):
    if mode == 'real':
        query_feats_main = torch.load(f'{directory}/query_feats_real_all.pth')
        query_feats_first = torch.load(f'{directory}/query_feats_real_first.pth')
        query_feats_second = torch.load(f'{directory}/query_feats_real_second.pth')
        gallery_feats_main = torch.load(f'{directory}/gallery_feats_real_all.pth')
        gallery_feats_first = torch.load(f'{directory}/gallery_feats_real_first.pth')
        gallery_feats_second = torch.load(f'{directory}/gallery_feats_real_second.pth')
    elif mode == 'real-aux':
        query_feats_main = torch.load(f'{directory}/query_feats_real-aux_all.pth')
        query_feats_first = torch.load(f'{directory}/query_feats_real-aux_first.pth')
        query_feats_second = torch.load(f'{directory}/query_feats_real-aux_second.pth')
        gallery_feats_main = torch.load(f'{directory}/gallery_feats_real-aux_all.pth')
        gallery_feats_first = torch.load(f'{directory}/gallery_feats_real-aux_first.pth')
        gallery_feats_second = torch.load(f'{directory}/gallery_feats_real-aux_second.pth')
    elif mode == 'real-fake':
        query_feats_main = torch.cat((
            torch.load(f'{directory}/query_feats_real_all.pth'),
            torch.load(f'{directory}/query_feats_fake_all.pth')
        ), dim=1)
        query_feats_first = torch.cat((
            torch.load(f'{directory}/query_feats_real_first.pth'),
            torch.load(f'{directory}/query_feats_fake_first.pth')
        ), dim=1)
        query_feats_second = torch.cat((
            torch.load(f'{directory}/query_feats_real_second.pth'),
            torch.load(f'{directory}/query_feats_fake_second.pth')
        ), dim=1)
        gallery_feats_main = torch.cat((
            torch.load(f'{directory}/gallery_feats_real_all.pth'),
            torch.load(f'{directory}/gallery_feats_fake_all.pth')
        ), dim=1)
        gallery_feats_first = torch.cat((
            torch.load(f'{directory}/gallery_feats_real_first.pth'),
            torch.load(f'{directory}/gallery_feats_fake_first.pth')
        ), dim=1)
        gallery_feats_second = torch.cat((
            torch.load(f'{directory}/gallery_feats_real_second.pth'),
            torch.load(f'{directory}/gallery_feats_fake_second.pth')
        ), dim=1)
    elif mode == 'real-fake-aux':
        query_feats_main = torch.cat((
            torch.load(f'{directory}/query_feats_real-aux_all.pth'),
            torch.load(f'{directory}/query_feats_fake-aux_all.pth')
        ), dim=1)
        query_feats_first = torch.cat((
            torch.load(f'{directory}/query_feats_real-aux_first.pth'),
            torch.load(f'{directory}/query_feats_fake-aux_first.pth')
        ), dim=1)
        query_feats_second = torch.cat((
            torch.load(f'{directory}/query_feats_real-aux_second.pth'),
            torch.load(f'{directory}/query_feats_fake-aux_second.pth')
        ), dim=1)
        gallery_feats_main = torch.cat((
            torch.load(f'{directory}/gallery_feats_real-aux_all.pth'),
            torch.load(f'{directory}/gallery_feats_fake-aux_all.pth')
        ), dim=1)
        gallery_feats_first = torch.cat((
            torch.load(f'{directory}/gallery_feats_real-aux_first.pth'),
            torch.load(f'{directory}/gallery_feats_fake-aux_first.pth')
        ), dim=1)
        gallery_feats_second = torch.cat((
            torch.load(f'{directory}/gallery_feats_real-aux_second.pth'),
            torch.load(f'{directory}/gallery_feats_fake-aux_second.pth')
        ), dim=1)
    return query_feats_main, query_feats_first, query_feats_second, \
           gallery_feats_main, gallery_feats_first, gallery_feats_second


if __name__ == '__main__':
    directory = opt.test_feat_path
    query_pids = torch.load(f'{directory}/query_pids.pth')
    query_modals = torch.load(f'{directory}/query_modals.pth')
    query_cids = torch.load(f'{directory}/query_cids.pth')
    gallery_pids = torch.load(f'{directory}/gallery_pids.pth')
    gallery_modals = torch.load(f'{directory}/gallery_modals.pth')
    gallery_cids = torch.load(f'{directory}/gallery_cids.pth')

    query_feats_main, query_feats_first, query_feats_second, \
    gallery_feats_main, gallery_feats_first, gallery_feats_second, \
        = get_features(mode='real-fake-aux')

    k_reciprocal = lambda x, y: K_Reciprocal(k1=5, k2=3, lambda_value=0)(x, y)

    lambda_1, lambda_2 = .8, .1

    start = time()
    for (q_modal, g_modal) in ((0, 0), (1, 1), (0, 1), (1, 0), (-1, -1)):
        if q_modal == -1:
            q_mask = query_modals >= q_modal
            g_mask = gallery_modals >= g_modal
        else:
            q_mask = query_modals == q_modal
            g_mask = gallery_modals == g_modal
        tmp_distance = euclidean_dist(query_feats_main[q_mask], gallery_feats_main[g_mask])
        '''re-ranking'''
        if lambda_1 != 0:
            tmp_distance_main = k_reciprocal(query_feats_main[q_mask], gallery_feats_main[g_mask])
            tmp_distance = tmp_distance * lambda_1 + tmp_distance_main * (1 - lambda_1)
        if lambda_2 != 0:
            tmp_distance_1to2 = k_reciprocal(query_feats_first[q_mask], gallery_feats_second[g_mask])
            tmp_distance_2to1 = k_reciprocal(query_feats_second[q_mask], gallery_feats_first[g_mask])
            tmp_distance += (tmp_distance_1to2 + tmp_distance_2to1) * lambda_2
        '''evaluate'''
        tmp_qid, tmp_gid = query_pids[q_mask], gallery_pids[g_mask]
        tmp_cmc, tmp_ap = evaluate(tmp_distance, tmp_qid, tmp_gid, opt)
        print_metrics(
            tmp_cmc, tmp_ap,
            prefix='{:<3}->{:<3}:  '.format(MODALITY_[q_modal], MODALITY_[g_modal])
        )
    # print(time() - start)