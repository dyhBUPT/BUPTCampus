"""
@Author: Du Yunhao
@Filename: train.py
@Contact: dyh_bupt@163.com
@Time: 2022/8/31 21:42
@Discription: train
"""
import os
import time
from itertools import cycle

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from utils import *
from opts import opt
from test import test
from loss import get_loss
from model import get_model
from dataloader import get_dataloader
from evaluation import evaluate, print_metrics
from torch.utils.tensorboard import SummaryWriter


os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
scaler = GradScaler()

save_configs(opt)
logger = get_logger(opt.save_dir)
writer = SummaryWriter(opt.save_dir)

dataloader_query, _ = get_dataloader(opt, 'query', False)
dataloader_gallery, _ = get_dataloader(opt, 'gallery', False)
dataloader_train, class_num = get_dataloader(opt, 'train', True)
dataloader_auxiliary, _ = get_dataloader(opt, 'auxiliary', False)

model = get_model(opt, class_num=class_num)

optimizer = eval(f'optim.{opt.optimizer}')(
    model.parameters(),
    lr=opt.base_lr,
    weight_decay=opt.weight_decay
)

loss_fn_tri = get_loss('triplet')
loss_fn_ce = get_loss('cross-entropy')

batch_size = 2 * opt.train_bs

if opt.resume_path:
    model, resume_epoch = load_from_ckpt(model, 'model', opt.resume_path)
else:
    resume_epoch = -1

print('========== Training ==========')
iteration = 0
logger.info('Start training!')
for epoch in range(resume_epoch+1, opt.max_epoch):
    model.train()
    LOSS_ID = AverageMeter('Loss(ID)', ':.4e')
    LOSS_TRI = AverageMeter('Loss(Tri)', ':.4e')
    LOSS_ID_AUX = AverageMeter('Loss(ID-AUX)', ':.4e')
    LOSS_TRI_AUX = AverageMeter('Loss(Tri-AUX)', ':.4e')
    BATCH_TIME = AverageMeter('Time', ':6.3f')
    lr = get_lr(opt, epoch)
    set_lr(optimizer, lr)
    meters = [BATCH_TIME, LOSS_TRI, LOSS_ID]
    PROGRESS = ProgressMeter(
        num_batches=len(dataloader_train),
        meters=meters,
        prefix="Epoch [{}/{}] ".format(epoch, opt.max_epoch),
        lr=lr
    )
    end = time.time()
    if opt.auxiliary:
        alpha = get_auxiliary_alpha(epoch, opt.max_epoch, phi=opt.aux_phi)
        for batch_idx, (datas, datas_aux) in enumerate(zip(dataloader_train, cycle(dataloader_auxiliary))):
            '''Auxiliary Set'''
            imgs_rgb_aux, imgs_ir_aux, labels_aux, cids_aux = datas_aux
            imgs_rgb_aux, imgs_ir_aux, labels_aux, cids_aux = \
                imgs_rgb_aux.cuda(), imgs_ir_aux.cuda(), labels_aux.cuda(), cids_aux.cuda()
            with autocast():
                feats_aux, logits_aux, labels_aux = model(imgs_rgb_aux, imgs_ir_aux, pids=labels_aux)
                loss_tri_aux = loss_fn_tri(
                    feats_aux,
                    labels_aux,
                    margin=opt.triplet_margin,
                    norm_feat=False,
                    hard_mining=opt.triplet_hard
                )
                LOSS_TRI_AUX.update(loss_tri_aux.item(), batch_size)
                loss_aux = alpha * loss_tri_aux
            '''Primary Set'''
            imgs_rgb, imgs_ir, labels, cids = datas
            imgs_rgb, imgs_ir, labels, cids = \
                imgs_rgb.cuda(), imgs_ir.cuda(), labels.cuda(), cids.cuda()
            with autocast():
                feats, logits, labels = model(imgs_rgb, imgs_ir, pids=labels)
                loss_tri = loss_fn_tri(
                    feats,
                    labels,
                    margin=opt.triplet_margin,
                    norm_feat=False,
                    hard_mining=opt.triplet_hard
                )
                LOSS_TRI.update(loss_tri.item(), batch_size)
                loss_id = loss_fn_ce(logits, labels)
                LOSS_ID.update(loss_id.item(), batch_size)
                loss = (1 - alpha) * (opt.lambda_tri * loss_tri + opt.lambda_ce * loss_id)
            '''Backward'''
            loss = loss + loss_aux
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            '''Write'''
            BATCH_TIME.update(time.time() - end)
            end = time.time()
            iteration += 1
            writer.add_scalar('Train/LR', lr, iteration)
            writer.add_scalar('Train/Alpha', alpha, iteration)
            writer.add_scalar('Loss/Triplet', loss_tri.item(), iteration)
            writer.add_scalar('Loss/Identity', loss_id.item(), iteration)
            writer.add_scalar('Loss/Auxiliary', loss_aux.item(), iteration)
            if batch_idx % opt.train_print_freq == 0:
                PROGRESS.display(batch_idx)
                logger.info(
                    'Epoch:[{}/{}] [{}/{}] Loss(Aux):{:.5f}'
                        .format(epoch, opt.max_epoch, batch_idx, len(dataloader_train), loss_aux.item())
                )
    else:
        for batch_idx, (imgs_rgb, imgs_ir, labels, cids) in enumerate(dataloader_train):
            '''Primary Set'''
            imgs_rgb, imgs_ir, labels, cids = \
                imgs_rgb.cuda(), imgs_ir.cuda(), labels.cuda(), cids.cuda()
            with autocast():
                feats, logits, labels = model(imgs_rgb, imgs_ir, pids=labels)
                loss_tri = loss_fn_tri(
                    feats,
                    labels,
                    margin=opt.triplet_margin,
                    norm_feat=False,
                    hard_mining=opt.triplet_hard
                )
                LOSS_TRI.update(loss_tri.item(), batch_size)
                loss_id = loss_fn_ce(logits, labels)
                LOSS_ID.update(loss_id.item(), batch_size)
                loss = opt.lambda_tri * loss_tri + opt.lambda_ce * loss_id
            '''Backward'''
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            '''Write'''
            BATCH_TIME.update(time.time() - end)
            end = time.time()
            iteration += 1
            writer.add_scalar('Loss/Triplet', loss_tri.item(), iteration)
            writer.add_scalar('Loss/Identity', loss_id.item(), iteration)
            if batch_idx % opt.train_print_freq == 0:
                PROGRESS.display(batch_idx)
                logger.info(
                    'Epoch:[{}/{}] [{}/{}] Loss(Tri):{:.5f}'
                        .format(epoch, opt.max_epoch, batch_idx, len(dataloader_train), loss_tri.item())
                )

    torch.cuda.empty_cache()
    if (epoch + 1) % opt.eval_freq == 0:
        CMC, MAP = test(model, dataloader_query, dataloader_gallery, show=True, return_all=True)
        writer.add_scalar('Eval/mAP(%)', MAP[-1], epoch)
        writer.add_scalar('Eval/Rank1(%)', CMC[-1][0], epoch)
        writer.add_scalar('Eval/Rank5(%)', CMC[-1][4], epoch)
        writer.add_scalar('Eval/Rank10(%)', CMC[-1][9], epoch)
        MODE = ['RGB->RGB', 'RGB->IR ', 'IR->RGB ', 'IR->IR  ', 'AllModal']
        log_info = 'Epoch:[{}/{}]'.format(epoch, opt.max_epoch)
        for i, mode in enumerate(MODE):
            log_info += '\n\t{}:  mAP:{:.2f}% Rank1:{:.2f}% Rank5:{:.2f}% Rank10:{:.2f}% Rank20:{:.2f}%'\
                .format(mode, MAP[i], CMC[i][0], CMC[i][4], CMC[i][9], CMC[i][19])
        logger.info(log_info)
    torch.cuda.empty_cache()

    if epoch + 1 >= 80:
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer,
            'epoch': epoch
        }
        torch.save(state_dict, join(opt.save_dir, f'epoch{epoch}.pth'))

logger.info('Finish training!')
