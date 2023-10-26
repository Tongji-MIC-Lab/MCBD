import argparse
import sys
import numpy as np
import pandas as pd
import json
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from lib import opts
from evaluation.eval_proposal import ANETproposal

num_prop = 1000


def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor


def Soft_NMS(df, alpha, nms_threshold=1e-5):
    df = df.sort_values(by="score", ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])
    rstart = []
    rend = []
    rscore = []

    for idx in range(0, len(tscore)):
        if tend[idx] - tstart[idx] >= 300:
            tscore[idx] = 0

    while len(tscore) > 1 and len(rscore) < num_prop and max(tscore)>0:
        max_index = tscore.index(max(tscore))
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
                tmp_width = tend[max_index] - tstart[max_index]
                if tmp_iou > alpha:
                    tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) / nms_threshold)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf


def _gen_detection_video(video_name, thu_label_id, opt):
    files = [opt['output']+"/results/" + f for f in os.listdir(opt['output']+"/results/") if
             video_name in f]
    if len(files) == 0:
        print('Missing result for video {}'.format(video_name))
    else:
        pass

    dfs = []
    for snippet_file in files:
        snippet_df = pd.read_csv(snippet_file)
        snippet_df['score'] = (snippet_df.start.values[:]*snippet_df.bstart.values[:]) * \
                              (snippet_df.end.values[:]*snippet_df.bend.values[:]) * \
                              (snippet_df.iou_clr.values[:]*snippet_df.iou_reg.values[:]*snippet_df.xc.values[:])**0.5
        dfs.append(snippet_df)
    df = pd.concat(dfs)
    if len(df) > 1:
        df = Soft_NMS(df, opt['alpha'], opt['nms_thr'])
    df = df.sort_values(by="score", ascending=False)

    fps = result[video_name]['fps']
    num_frames = result[video_name]['num_frames']
    proposal_list = []
    for j in range(min(num_prop, len(df))):
        tmp_proposal = {}
        tmp_proposal["score"] = float(round(df.score.values[j], 6))
        tmp_proposal["segment"] = [float(round(max(0, df.xmin.values[j]) / fps, 1)),
                                   float(round(min(num_frames, df.xmax.values[j]) / fps, 1))]
        proposal_list.append(tmp_proposal)
    return {video_name:proposal_list}


def gen_detection_multicore(opt):
    thumos_test_anno = pd.read_csv("./data/thumos_annotations/test_Annotation.csv")
    video_list = thumos_test_anno.video.unique()
    thu_label_id = np.sort(thumos_test_anno.type_idx.unique())[1:] - 1
    thu_video_id = np.array([int(i[-4:]) - 1 for i in video_list])
    cls_data = np.load("./data/uNet_test.npy")
    cls_data = cls_data[thu_video_id,:][:, thu_label_id]

    thumos_gt = pd.read_csv("./data/thumos_annotations/thumos14_test_groundtruth.csv")
    global result
    result = {
        video:
            {
                'fps': thumos_gt.loc[thumos_gt['video-name'] == video]['frame-rate'].values[0],
                'num_frames': thumos_gt.loc[thumos_gt['video-name'] == video]['video-frames'].values[0]
            }
        for video in video_list
    }
    parallel = Parallel(n_jobs=16, prefer="processes")
    detection = parallel(delayed(_gen_detection_video)(video_name, thu_label_id, opt)
                        for video_name in tqdm(video_list, ncols=40))

    detection_dict = {}
    [detection_dict.update(d) for d in detection]
    output_dict = {"version": "THUMOS14", "results": detection_dict, "external_data": {}}
    print('dumping',opt["output"] + '/proposal_result.json')
    with open(opt["output"] + '/proposal_result.json', "w") as out:
        json.dump(output_dict, out)


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


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    print('alpha:',opt['alpha'], 'nms_thr:',opt['nms_thr'])

    if not os.path.exists(opt["output"]):
        os.makedirs(opt["output"])

    print("Proposal post-processing start")
    gen_detection_multicore(opt)
    print("Proposal post-processing finished")

    prediction_filename = opt["output"] + '/proposal_result.json'
    ground_truth_filename = 'data/annotations/thumos14.json'
    uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid = \
        run_evaluation(
            ground_truth_filename,
            prediction_filename,
            max_avg_nr_proposals=1000,
            tiou_thresholds=np.linspace(0.5, 1.0, 11),
            subset='test')

    ANs = [50,100,200,500,1000]
    for i in ANs:
        print('AR@', i, 'is \t', np.mean(uniform_recall_valid[:, i-1]))