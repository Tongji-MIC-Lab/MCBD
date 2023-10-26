import os
import torch
import torch.nn.parallel
import torch.optim as optim
from torch import autograd
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import tqdm
from lib import opts
from lib.model import MCBD
from lib.dataset import VideoDataSet
import random

seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
opt = opts.parse_opt()
opt = vars(opt)
if opt['rgb']:
    checkpoint_dir = opt["output"] +'_rgb'
else:
    checkpoint_dir = opt["output"] +'_flow'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

batch_size = opt["batch_size"]
tscale = opt["temporal_scale"]
feature_dim = opt["feat_dim"]


def gen_mask(tscale):
    mask = np.zeros([tscale, tscale], np.float32)
    for i in range(tscale):
        for j in range(i, tscale):
            mask[i, j] = 1
    return mask


mask = gen_mask(tscale)
mask = np.expand_dims(np.expand_dims(mask, 0), 1)
mask = torch.from_numpy(mask).float().requires_grad_(False).cuda()
tmp_mask = mask.repeat(batch_size, 1, 1, 1).requires_grad_(False)
tmp_mask = tmp_mask > 0


def bmn_loss_func(pred_bm, gt_iou_map, bm_mask):
    pred_bm_reg = pred_bm[:, 0].contiguous()
    pred_bm_cls = pred_bm[:, 1].contiguous()
    gt_iou_map = gt_iou_map * bm_mask
    pem_reg_loss = pem_reg_loss_func(pred_bm_reg, gt_iou_map, bm_mask)
    pem_cls_loss = pem_cls_loss_func(pred_bm_cls, gt_iou_map, bm_mask)
    loss = 10 * pem_reg_loss + pem_cls_loss
    return loss, pem_reg_loss, pem_cls_loss


def pem_reg_loss_func(pred_score, gt_iou_map, mask):
    u_hmask = (gt_iou_map > 0.7).float()
    u_mmask = ((gt_iou_map <= 0.7) & (gt_iou_map > 0.3)).float()
    u_lmask = ((gt_iou_map <= 0.3) & (gt_iou_map > 0.)).float()
    u_lmask = u_lmask * mask
    num_h = torch.sum(u_hmask)
    num_m = torch.sum(u_mmask)
    num_l = torch.sum(u_lmask)
    r_m = num_h / num_m
    u_smmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_smmask = u_mmask * u_smmask
    u_smmask = (u_smmask > (1. - r_m)).float()
    r_l = num_h / num_l
    u_slmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_slmask = u_lmask * u_slmask
    u_slmask = (u_slmask > (1. - r_l)).float()
    weights = u_hmask + u_smmask + u_slmask
    loss = F.smooth_l1_loss(pred_score * weights, gt_iou_map * weights)
    loss = 0.5 * torch.sum(loss * torch.ones(*weights.shape).cuda()) / torch.sum(weights)

    return loss


def pem_cls_loss_func(pred_score, gt_iou_map, mask):
    pmask = (gt_iou_map > 0.9).float()
    nmask = (gt_iou_map <= 0.9).float()
    nmask = nmask * mask
    num_positive = torch.sum(pmask)
    num_entries = num_positive + torch.sum(nmask)
    ratio = num_entries / num_positive
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 0.000001
    loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
    loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * nmask
    loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
    return loss


def binary_logistic_loss(gt_scores, pred_anchors):
    gt_scores = gt_scores.view(-1)
    pred_anchors = pred_anchors.view(-1)
    pmask = (gt_scores > 0.5).float()
    num_positive = torch.sum(pmask)
    num_entries = pmask.size()[0]
    ratio = num_entries / max(num_positive, 1)
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 1e-6
    neg_pred_anchors = 1.0 - pred_anchors + epsilon
    pred_anchors = pred_anchors + epsilon
    loss = coef_1 * pmask * torch.log(pred_anchors) + coef_0 * (1.0 - pmask) * torch.log(
        neg_pred_anchors)
    loss = -1.0 * torch.mean(loss)
    return loss


def train(net, dl_iter, optimizer, epoch, training):
    if training:
        net.train()
    else:
        net.eval()
    loss_action_val = 0
    loss_b_start_val = 0
    loss_b_end_val = 0
    loss_iou_val = 0
    loss_start_val = 0
    loss_end_val = 0
    cost_val = 0
    n_iter = 0
    dl_iter = tqdm.tqdm(dl_iter, ncols=40)
    for n_iter, \
        (gt_action, gt_start, gt_end, feature, iou_label, gt_boundary) in enumerate(dl_iter):
        gt_action = gt_action.cuda()
        gt_start = gt_start.cuda()
        gt_end = gt_end.cuda()
        feature = feature.cuda()
        iou_label = iou_label.cuda()
        gt_boundary = gt_boundary.cuda()

        output_dict = net(feature)
        xc = output_dict['xc']
        xb_start = output_dict['xb_start']
        xb_end = output_dict['xb_end']
        iou = output_dict['iou']
        prop_start = output_dict['prop_start']
        prop_end = output_dict['prop_end']

        loss_action = binary_logistic_loss(gt_action, xc)
        loss_b_start = binary_logistic_loss(gt_start, xb_start)
        loss_b_end = binary_logistic_loss(gt_end, xb_end)
        loss_iou, pem_reg_loss, pem_cls_loss = bmn_loss_func(iou, iou_label.squeeze(1), mask)

        gt_start = torch.unsqueeze(gt_start, 1)
        gt_end = torch.unsqueeze(gt_end, 1)
        gt_start = torch.unsqueeze(gt_start, 3).repeat(1, 1, 1, tscale)
        gt_end = torch.unsqueeze(gt_end, 2).repeat(1, 1, tscale, 1)
        loss_start = binary_logistic_loss(
            torch.masked_select(gt_start, tmp_mask),
            torch.masked_select(prop_start, tmp_mask)
        )
        loss_end = binary_logistic_loss(
            torch.masked_select(gt_end, tmp_mask),
            torch.masked_select(prop_end, tmp_mask)
        )
        cost = loss_action + loss_b_start+loss_b_end + loss_iou + loss_start + loss_end

        if training:
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        loss_action_val += loss_action.cpu().detach().numpy()
        loss_b_start_val += loss_b_start.cpu().detach().numpy()
        loss_b_end_val += loss_b_end.cpu().detach().numpy()
        loss_iou_val += loss_iou.cpu().detach().numpy()
        loss_start_val += loss_start.cpu().detach().numpy()
        loss_end_val += loss_end.cpu().detach().numpy()
        cost_val += cost.cpu().detach().numpy()

    loss_action_val /= (n_iter + 1)
    loss_b_start_val /= (n_iter + 1)
    loss_b_end_val /= (n_iter + 1)
    loss_iou_val /= (n_iter + 1)
    loss_start_val /= (n_iter + 1)
    loss_end_val /= (n_iter + 1)
    cost_val /= (n_iter + 1)

    if training:
        print(
            "Epoch-%d Train Loss: "
            "Total - %.05f, Action - %.05f, bstart - %.05f, bend - %.05f, Start - %.05f, End - %.05f, IoU - %.05f"
            % (epoch, cost_val, loss_action_val, loss_b_start_val, loss_b_end_val, loss_start_val, loss_end_val, loss_iou_val))
        torch.save(net.module.state_dict(),
                   os.path.join(checkpoint_dir, 'checkpoint-%d.ckpt' % epoch))
    else:
        print(
            "Epoch-%d Validation Loss: "
            "Total - %.05f, Action - %.05f, bstart - %.05f, bend - %.05f, Start - %.05f, End - %.05f, IoU - %.05f"
            % (epoch, cost_val, loss_action_val, loss_b_start_val, loss_b_end_val, loss_start_val, loss_end_val, loss_iou_val))

        if cost_val < net.module.best_loss:
            net.module.best_loss = cost_val
            torch.save(net.module.state_dict(),
                       os.path.join(checkpoint_dir, 'checkpoint-best.ckpt'))


if __name__ == '__main__':
    model = MCBD(feature_dim, opt['rgb'])
    model = torch.nn.DataParallel(model, device_ids=list(range(opt['n_gpu']))).cuda()
    print('use {} gpus to train!'.format(opt['n_gpu']))

    optimizer = optim.Adam(model.parameters(), lr=opt["training_lr"],
                           weight_decay=opt["weight_decay"])
    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8,  drop_last=True, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=opt["batch_size"], shuffle=False,
                                              num_workers=8,  drop_last=True, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])
    for epoch in range(opt["train_epochs"]):
      with autograd.detect_anomaly():
        print('current learning rate:', scheduler.get_lr()[0])
        train(model, train_loader, optimizer, epoch, training=True)
        train(model, test_loader, optimizer, epoch, training=False)
        scheduler.step()





