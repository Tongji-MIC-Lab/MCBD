import argparse
import json
import multiprocessing as mp
import os
import threading
import numpy as np
import pandas as pd
import tqdm
from utils import getDatasetDict
from lib.config_loader import config


parser = argparse.ArgumentParser()
parser.add_argument('--top_number', type=int, nargs='?', default=100)
parser.add_argument('-t', '--thread', type=int, nargs='?', default=8)
parser.add_argument('-m', '--mode', type=str, nargs='?', default='validation')

parser.add_argument('--alpha', type=float, default=0.65)
parser.add_argument('--beta', type=float, default=0.27)
parser.add_argument('--nms_threshold', type=float, default=0.3)

args = parser.parse_args()
top_number = args.top_number
thread_num = args.thread
alpha = args.alpha
beta = args.beta
nms_threshold = args.nms_threshold


def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor


def softNMS(df):
    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])

    rstart = []
    rend = []
    rscore = []
    while len(tscore) > 1 and len(rscore) < top_number:
        max_index = tscore.index(max(tscore))
        tmp_start = tstart[max_index]
        tmp_end = tend[max_index]
        tmp_score = tscore[max_index]
        rstart.append(tmp_start)
        rend.append(tmp_end)
        rscore.append(tmp_score)
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)
        tstart = np.array(tstart)
        tend = np.array(tend)
        tscore = np.array(tscore)
        tt1 = np.maximum(tmp_start, tstart)
        tt2 = np.minimum(tmp_end, tend)
        intersection = tt2 - tt1
        duration = tend - tstart
        tmp_width = tmp_end - tmp_start
        iou = intersection / (tmp_width + duration - intersection).astype(float)
        idxs = np.where(iou > alpha + beta * tmp_width)[0]
        tscore[idxs] = tscore[idxs] * np.exp(-np.square(iou[idxs]) / nms_threshold)
        tstart = list(tstart)
        tend = list(tend)
        tscore = list(tscore)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf


def sub_processor(lock, pid, video_list):
    for i in range(len(video_list)):
        video_name = video_list[i]
        df = pd.read_csv(os.path.join(result_dir, video_name + ".csv"))
        df['score'] = (df.start.values[:] * df.bstart.values[:]) * \
                    (df.end.values[:] * df.bend.values[:]) * \
                    (df.iou_clr.values[:]* df.iou_reg.values[:]*df.xc.values[:])**0.5
        if len(df) > 1:
            df = softNMS(df)
        df = df.sort_values(by="score", ascending=False)
        video_info = video_dict[video_name]
        video_duration = video_info["duration_second"]
        proposal_list = []
        for j in range(min(top_number, len(df))):
            tmp_proposal = {}
            tmp_proposal["score"] = df.score.values[j]
            tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
                                       min(1, df.xmax.values[j]) * video_duration]
            proposal_list.append(tmp_proposal)
        result_dict[video_name[2:]] = proposal_list


video_info_file = 'data/video_info_19993.json'
train_dict, val_dict, test_dict = getDatasetDict(video_info_file)
mode = args.mode
if mode == 'validation':
    video_dict = val_dict
else:
    video_dict = test_dict
result_dir = config.result_dir
output_file = config.result_dir+'/result_proposals.json'
video_list = list(video_dict.keys())

global result_dict
result_dict = mp.Manager().dict()
processes = []
lock = threading.Lock()
total_video_num = len(video_list)
per_thread_video_num = total_video_num // thread_num
for i in range(thread_num):
    if i == thread_num - 1:
        sub_video_list = video_list[i * per_thread_video_num:]
    else:
        sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
    p = mp.Process(target=sub_processor, args=(lock, i, sub_video_list))
    p.start()
    processes.append(p)
for p in processes:
    p.join()

result_dict = dict(result_dict)
output_dict = {"version": "VERSION 1.3", "results": result_dict, "external_data": {}}
with open(output_file, 'w') as outfile:
    json.dump(output_dict, outfile)


from evaluation.eval_proposal import ANETproposal

def run_evaluation(ground_truth_filename, proposal_filename,
                   max_avg_nr_proposals=100,
                   tiou_thresholds=np.linspace(0.5, 0.95, 10),
                   subset='validation'):
    anet_proposal = ANETproposal(ground_truth_filename, proposal_filename,
                                 tiou_thresholds=tiou_thresholds,
                                 max_avg_nr_proposals=max_avg_nr_proposals,
                                 subset=subset, verbose=True, check_status=False)
    anet_proposal.evaluate()
    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    average_nr_proposals = anet_proposal.proposals_per_video
    return (average_nr_proposals, average_recall, recall)


eval_file = output_file
json_name = 'data/activity_net_1_3_new.json'
uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid = \
    run_evaluation(
        json_name,
        eval_file,
        max_avg_nr_proposals=100,
        tiou_thresholds=np.linspace(0.5, 0.95, 10),
        subset='validation')

print("AR@1 is \t", np.mean(uniform_recall_valid[:, 0]))
print("AR@5 is \t", np.mean(uniform_recall_valid[:, 4]))
print("AR@10 is \t", np.mean(uniform_recall_valid[:, 9]))
print("AR@100 is \t", np.mean(uniform_recall_valid[:, -1]))