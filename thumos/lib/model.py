import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init


class x_1d(torch.nn.Module):
    def __init__(self, feature_dim, rgb):
        super(x_1d, self).__init__()
        self.feat_dim = feature_dim
        self.rgb = rgb
        self.temporal_dim = 100
        self.c_hidden = 512
        self.output_dim = 3
        self.conv1 = torch.nn.Conv1d(in_channels=self.feat_dim//2,    out_channels=self.c_hidden,kernel_size=3,stride=1,padding=1,groups=1)
        self.conv2 = torch.nn.Conv1d(in_channels=self.c_hidden,out_channels=128,kernel_size=3,stride=1,padding=1,groups=1)
        self.conv3 = torch.nn.Conv1d(in_channels=128,out_channels=self.output_dim,   kernel_size=1,stride=1,padding=0)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        if self.rgb:
            x = x[:, : self.feat_dim//2]
        else:
            x = x[:, self.feat_dim//2: ]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x4 = torch.sigmoid(0.01*self.conv3(x))
        xb_start, xb_end, xc = torch.split(x4, 1, 1)
        output_dict = {
            'xc_feat': x,
            'xc': xc,
            'xb_start': xb_start,
            'xb_end': xb_end,
        }
        return output_dict


class MCBD(nn.Module):

    def __init__(self, feature_dim, rgb):
        super(MCBD, self).__init__()
        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512
        self.num_sample = 32
        self.tscale = 128
        self.prop_boundary_ratio = 0.5
        self.num_sample_perbin = 3
        self._get_interp1d_mask()

        self.x_1d = x_1d(feature_dim, rgb)

        self.best_loss = 999999
        
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(128, 512, kernel_size=(self.num_sample, 1, 1),stride=(self.num_sample, 1, 1)),
            nn.ReLU(inplace=True)
        )

        self.x_2d_p = nn.Sequential(
            nn.Conv2d(512, self.hidden_dim_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, kernel_size=1),
            nn.Sigmoid())


    def forward(self, x):
        frame_feat = self.x_1d(x)
        confidence_map = self._boundary_matching_layer(frame_feat['xc_feat'])
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)
        prop_feat = self.x_2d_p(confidence_map)
        prop_start = prop_feat[:, 0:1].contiguous()
        prop_end = prop_feat[:, 1:2].contiguous()
        iou = prop_feat[:, 2:].contiguous()

        output_dict = {
            'xc': frame_feat['xc'],
            'xb_start': frame_feat['xb_start'],
            'xb_end': frame_feat['xb_end'],
            'iou': iou,
            'prop_start': prop_start,
            'prop_end': prop_end
        }
        return output_dict

    def _boundary_matching_layer(self, x):
        input_size = x.size()
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0],input_size[1],self.num_sample,self.tscale,self.tscale)
        return out

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 0.5
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += 0.5
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for end_index in range(self.tscale):
            mask_mat_vector = []
            for start_index in range(self.tscale):
                if start_index <= end_index:
                    p_xmin = start_index
                    p_xmax = end_index + 1
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.tscale, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False)
