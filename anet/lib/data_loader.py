import torch
from torch.utils.data import Dataset
from lib.config_loader import config
from utils import *


tscale = config.tscale
tgap = 1.0 / tscale
video_info_file = config.video_info_file
video_filter = config.video_filter
data_dir = config.feat_dir
data_aug = config.data_aug


class VideoDataSet(Dataset):
    def __init__(self, mode='training'):
        train_dict, val_dict, test_dict = getDatasetDict(video_info_file, video_filter)
        training = True
        if mode == 'training':
            video_dict = train_dict
        else:
            training = False
            video_dict = val_dict
        self.mode = mode
        self.video_dict = video_dict
        video_num = len(list(video_dict.keys()))
        video_list = np.arange(video_num)
        if training:
            data_dict, train_video_mean_len = getFullData(video_dict, config,
                                                          last_channel=False,
                                                          training=True)
        else:
            data_dict = getFullData(video_dict, config,
                                    last_channel=False, training=False)
        for key in list(data_dict.keys()):
            data_dict[key] = torch.Tensor(data_dict[key]).float()
        self.data_dict = data_dict
        if data_aug and training:
            add_list = np.where(np.array(train_video_mean_len) < 0.2)
            add_list = np.reshape(add_list, [-1])
            video_list = np.concatenate([video_list, add_list[:]], 0)
        self.video_list = video_list
        np.random.shuffle(self.video_list)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self.video_list[idx]
        data_dict = self.data_dict
        gt_action = data_dict['gt_action'][idx].unsqueeze(0)
        gt_start = data_dict['gt_start'][idx].unsqueeze(0)
        gt_end = data_dict['gt_end'][idx].unsqueeze(0)
        feature = data_dict['feature'][idx]
        iou_label = data_dict['iou_label'][idx].unsqueeze(0)

        return gt_action, gt_start, gt_end, feature, iou_label
