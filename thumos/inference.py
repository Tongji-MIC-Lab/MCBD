import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm

from lib import opts
from lib.model import MCBD
from lib.dataset import VideoDataSet
from lib import opts

opt = opts.parse_opt()
opt = vars(opt)

if not os.path.exists(opt["output"]+ "/results"):
    os.makedirs(opt["output"]+ "/results")

checkpoint_dir_rgb = opt["output"] +'_rgb'
checkpoint_dir_flow = opt["output"] +'_flow'

if not os.path.exists(checkpoint_dir_rgb + "/results"):
    os.makedirs(checkpoint_dir_rgb + "/results")
if not os.path.exists(checkpoint_dir_flow + "/results"):
    os.makedirs(checkpoint_dir_flow + "/results")

tscale = opt["temporal_scale"]
feature_dim = opt["feat_dim"]
skip = opt['skip_videoframes']
ckpt = opt['checkpoint']
ckpt_flow = opt['checkpoint_flow']

def gen_mask(tscale):
    """
    generator map mask
    :param tscale: temporal scale of feature
    :return: numpy array
    """
    mask = np.zeros([tscale, tscale], np.float32)
    for i in range(tscale):
        for j in range(i, tscale):
            mask[i, j] = 1
    return mask

mask = gen_mask(tscale)
mask = np.expand_dims(np.expand_dims(mask, 0), 1)
mask = torch.from_numpy(mask).float().requires_grad_(False).cuda()

if __name__ == '__main__':

    net_rgb = MCBD(feature_dim, True)
    state_dict_rgb = torch.load(os.path.join(checkpoint_dir_rgb, ckpt))
    print('load model '+ os.path.join(checkpoint_dir_rgb, ckpt))
    net_rgb.load_state_dict(state_dict_rgb)
    net_rgb = nn.DataParallel(net_rgb, device_ids=[0]).cuda()
    # net_rgb.to(torch.device("cuda:0" ))
    net_rgb.eval()

    net_flow = MCBD(feature_dim, False)
    state_dict_flow = torch.load(os.path.join(checkpoint_dir_flow, ckpt_flow))
    print('load model '+ os.path.join(checkpoint_dir_flow, ckpt_flow))
    net_flow.load_state_dict(state_dict_flow)
    net_flow = nn.DataParallel(net_flow, device_ids=[0]).cuda()
    # net_flow.to(torch.device("cuda:1" ))
    net_flow.eval()

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation", mode='inference'),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)

    print("Inference start")
    with torch.no_grad():
        for idx, input_data in tqdm.tqdm(test_loader, ncols=40):
            video_name = test_loader.dataset.video_list[idx[0]]
            offset = min(test_loader.dataset.data['indices'][idx[0]])
            video_name = video_name+'_{}'.format(math.floor(offset/250))
            input_data = input_data.cuda()

            # rgb
            output_dict_rgb = net_rgb(input_data)
            out_iou_rgb = output_dict_rgb['iou']
            out_start_rgb = output_dict_rgb['prop_start']
            out_end_rgb = output_dict_rgb['prop_end']
            out_bstart_rgb = output_dict_rgb['xb_start']
            out_bend_rgb = output_dict_rgb['xb_end']
            out_xc_rgb = output_dict_rgb['xc']
            # fusion starting and ending map score
            out_start_rgb = out_start_rgb * mask
            out_end_rgb = out_end_rgb * mask
            out_start_rgb = torch.sum(out_start_rgb, 3) / torch.sum(mask, 3)
            out_end_rgb = torch.sum(out_end_rgb, 2) / torch.sum(mask, 2)
            out_iou_clr_rgb = out_iou_rgb[:, 0].cpu().detach().numpy()[0]
            out_iou_reg_rgb = out_iou_rgb[:, 1].cpu().detach().numpy()[0]
            out_start_rgb = out_start_rgb[:, 0].cpu().detach().numpy()[0]
            out_end_rgb = out_end_rgb[:, 0].cpu().detach().numpy()[0]
            out_bstart_rgb = out_bstart_rgb[:,0].cpu().detach().numpy()[0]
            out_bend_rgb = out_bend_rgb[:,0].cpu().detach().numpy()[0]
            out_xc_rgb = out_xc_rgb[:,0].cpu().detach().numpy()[0]


            # flow
            output_dict_flow = net_flow(input_data)
            out_iou_flow = output_dict_flow['iou']
            out_start_flow = output_dict_flow['prop_start']
            out_end_flow = output_dict_flow['prop_end']
            out_bstart_flow = output_dict_flow['xb_start']
            out_bend_flow = output_dict_flow['xb_end']
            out_xc_flow = output_dict_flow['xc']
            # fusion starting and ending map score
            out_start_flow = out_start_flow * mask
            out_end_flow = out_end_flow * mask
            out_start_flow = torch.sum(out_start_flow, 3) / torch.sum(mask, 3)
            out_end_flow = torch.sum(out_end_flow, 2) / torch.sum(mask, 2)
            out_iou_clr_flow = out_iou_flow[:, 0].cpu().detach().numpy()[0]
            out_iou_reg_flow = out_iou_flow[:, 1].cpu().detach().numpy()[0]
            out_start_flow = out_start_flow[:, 0].cpu().detach().numpy()[0]
            out_end_flow = out_end_flow[:, 0].cpu().detach().numpy()[0]
            out_bstart_flow = out_bstart_flow[:,0].cpu().detach().numpy()[0]
            out_bend_flow = out_bend_flow[:,0].cpu().detach().numpy()[0]
            out_xc_flow = out_xc_flow[:,0].cpu().detach().numpy()[0]

            # fusion
            out_iou_clr = (out_iou_clr_rgb + out_iou_clr_flow) / 2
            out_iou_reg = (out_iou_reg_rgb + out_iou_reg_flow) / 2
            out_start = (out_start_rgb + out_start_flow) / 2
            out_end = (out_end_rgb + out_end_flow) / 2
            out_bstart = (out_bstart_rgb + out_bstart_flow) / 2
            out_bend = (out_bend_rgb + out_bend_flow) / 2
            out_xc = (out_xc_rgb + out_xc_flow) / 2

            new_props = []
            tmp_xmin = [skip * i for i in range(tscale)] + offset
            tmp_xmax = [skip * i for i in range(1, tscale + 1)]+ offset
            for i in range(tscale):
                for j in range(i, tscale):
                    start = out_start[i]
                    end = out_end[j]
                    iou_clr = out_iou_clr[i, j]
                    iou_reg = out_iou_reg[i, j]
                    bstart, bend = out_bstart[i], out_bend[j]
                    if i==j:
                        xc=0.0
                    else:
                        xc = np.mean(out_xc[i:j])
                    new_props.append([tmp_xmin[i], tmp_xmax[j], bstart, bend, xc, start, end, iou_clr, iou_reg])

            new_props = np.stack(new_props)
            col_name = ["xmin", "xmax","bstart","bend","xc","start", "end", "iou_clr", "iou_reg"]
            new_df = pd.DataFrame(new_props, columns=col_name)
            new_df.to_csv(opt["output"]+"/results/" + video_name + ".csv", index=False)

    print("Inference finished")
