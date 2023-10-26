import argparse
import os
import numpy as np
import tqdm
from lib import data_loader
import utils
from lib.model import MCBD
from lib.config_loader import config
import torch
import torch.nn as nn
import time

checkpoint_dir = config.checkpoint_dir
result_dir = config.result_dir
tscale = config.tscale
feature_dim = config.feature_dim
batch_size = config.test_batch_size
test_mode = config.test_mode
mask = data_loader.gen_mask(tscale)
mask = np.expand_dims(np.expand_dims(mask, 0), 1)
mask = torch.from_numpy(mask).float().requires_grad_(False).cuda()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoint-best.ckpt')
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    with torch.no_grad():
        net = MCBD(feature_dim)
        state_dict = torch.load(os.path.join(checkpoint_dir, args.checkpoint))
        net.load_state_dict(state_dict)
        net = nn.DataParallel(net, device_ids=[0]).cuda()
        net.eval()
        train_dict, val_dict, test_dict = data_loader.getDatasetDict(config.video_info_file)
        if test_mode == 'validation':
            video_dict = val_dict
        else:
            video_dict = test_dict

        batch_video_list = data_loader.getBatchListTest(video_dict, batch_size)

        batch_result_xmin = []
        batch_result_xmax = []
        batch_result_iou_clr = []
        batch_result_iou_reg = []
        batch_result_pstart = []
        batch_result_pend = []
        batch_result_bstart = []
        batch_result_bend = []
        batch_result_xc = []

        for idx in tqdm.tqdm(range(len(batch_video_list)), ncols=40):
            batch_anchor_xmin, batch_anchor_xmax, batch_anchor_feature = \
                data_loader.getProposalDataTest(batch_video_list[idx], config)
            in_feature = torch.from_numpy(batch_anchor_feature).float().cuda().permute(0, 2, 1)

            
            # tim = 0
            # start = torch.cuda.Event(enable_timing=True) #the times
            # end = torch.cuda.Event(enable_timing=True)
            # start.record()
            # # out = net(image_input)
            # output_dict = net(in_feature)
            
            # end.record()
            # torch.cuda.synchronize()
            # tim = start.elapsed_time(end)
            # print(tim)


            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)

            # start.record(stream=torch.cuda.current_stream())
            # # output_tensor = model(input_tensor)
            # output_dict = net(in_feature)
            # end.record(stream=torch.cuda.current_stream())
            # end.synchronize()
            # tim = start.elapsed_time(end)
            # print(tim)

            torch.cuda.synchronize()
            start = time.time()
            output_dict = net(in_feature)
            torch.cuda.synchronize()
            end = time.time()
            total_time = end - start
            # print('total_time:{:.3f}'.format(total_time))
            # for batch_size 1
            # total_time:0.010

            out_iou = output_dict['iou']
            out_start = output_dict['prop_start']
            out_end = output_dict['prop_end']
            out_bstart = output_dict['xb_start']
            out_bend = output_dict['xb_end']
            out_xc = output_dict['xc']

            out_start = out_start * mask
            out_end = out_end * mask
            out_start = torch.sum(out_start, 3) / torch.sum(mask, 3)
            out_end = torch.sum(out_end, 2) / torch.sum(mask, 2)

            batch_result_xmin.append(batch_anchor_xmin)
            batch_result_xmax.append(batch_anchor_xmax)
            batch_result_iou_clr.append(out_iou[:, 0].cpu().detach().numpy())
            batch_result_iou_reg.append(out_iou[:, 1].cpu().detach().numpy())
            batch_result_pstart.append(out_start[:, 0].cpu().detach().numpy())
            batch_result_pend.append(out_end[:, 0].cpu().detach().numpy())
            batch_result_bstart.append(out_bstart[:,0].cpu().detach().numpy())
            batch_result_bend.append(out_bend[:,0].cpu().detach().numpy())
            batch_result_xc.append(out_xc[:,0].cpu().detach().numpy())

        utils.save_proposals_result(batch_video_list,
                                    batch_result_xmin,
                                    batch_result_xmax,
                                    batch_result_iou_clr,batch_result_iou_reg,
                                    batch_result_pstart,
                                    batch_result_pend,
                                    batch_result_bstart,batch_result_bend,batch_result_xc,
                                    tscale, result_dir)
