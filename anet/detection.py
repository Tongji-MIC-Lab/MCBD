import sys
import argparse
import numpy as np
import pandas as pd
import json
import os
from joblib import Parallel, delayed
import tqdm
from lib.config_loader import config

result_dir = config.result_dir+'/'
prediction_filename = result_dir+ 'detection_result_nms.json'


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def get_infer_dict():
    video_info_path = "./data/activitynet_annotations/video_info_new.csv"
    video_anno_path = "./data/activitynet_annotations/anet_anno_action.json"
    df = pd.read_csv(video_info_path)
    json_data = load_json(video_anno_path)
    database = json_data
    video_dict = {}
    for i in range(len(df)):
        video_name = df.video.values[i]
        video_info = database[video_name]
        video_new_info = {}
        video_new_info['duration_frame'] = video_info['duration_frame']
        video_new_info['duration_second'] = video_info['duration_second']
        video_new_info["feature_frame"] = video_info['feature_frame']
        video_subset = df.subset.values[i]
        video_new_info['annotations'] = video_info['annotations']
        if video_subset == 'validation':
            video_dict[video_name] = video_new_info
    del video_dict['v_0dkIbKXXFzI']
    return video_dict


def Soft_NMS(df, nms_threshold=0.35, num_prop=100):
    df = df.sort_values(by="score", ascending=False)
    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])
    rstart = []
    rend = []
    rscore = []
    while len(tscore) > 1 and len(rscore) < num_prop and max(tscore)>0:
        max_index = tscore.index(max(tscore))
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
                tmp_width = tend[max_index] - tstart[max_index]
                if tmp_iou > 0:
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


def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor


def _gen_detection_video(video_name, video_score, video_cls, video_info, num_prop=200, topk = 2):
    
    score_1 = np.max(video_score)
    class_1 = video_cls[np.argmax(video_score)]
    video_score[np.argmax(video_score)] = -1
    score_2 = np.max(video_score)
    class_2 = video_cls[np.argmax(video_score)]
    df = pd.read_csv(result_dir + "/v_" + video_name + ".csv")
    df['score'] = df.iou_clr.values[:] * df.iou_reg.values[:] * df.start.values[:] * df.end.values[:]
    if len(df) > 1:
        df = Soft_NMS(df)
    df = df.sort_values(by="score", ascending=False)
    video_duration=float((video_info["duration_frame"]/16)*16)/video_info["duration_frame"]*video_info["duration_second"]
    proposal_list = []
    for j in range(min(100, len(df))):
        tmp_proposal = {}
        tmp_proposal["label"] = str(class_1)
        tmp_proposal["score"] = float(df.score.values[j] * score_1)
        tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
                                   min(1, df.xmax.values[j]) * video_duration]
        proposal_list.append(tmp_proposal)
    return {video_name: proposal_list}


def gen_detection_multicore():
    infer_dict = get_infer_dict()
    cls_data = load_json("data/cuhk_val_simp_share.json")
    cls_data_score, cls_data_cls = {}, {}
    for idx, vid in enumerate(infer_dict.keys()):
        vid = vid[2:]
        cls_data_score[vid] = np.array(cls_data["results"][vid])
        cls_data_cls[vid] = cls_data["class"]
    parallel = Parallel(n_jobs=8, prefer="processes")
    detection = parallel(delayed(_gen_detection_video)(vid, cls_data_score[vid], video_cls, infer_dict['v_'+vid])
                        for vid, video_cls in tqdm.tqdm(cls_data_cls.items(), ncols=40))
    detection_dict = {}
    [detection_dict.update(d) for d in detection]
    output_dict = {"version": "ANET v1.3, GTAD", "results": detection_dict, "external_data": {}}
    with open(prediction_filename, "w") as out:
        json.dump(output_dict, out)


if __name__ == "__main__":

    print("Detection post processing start")
    gen_detection_multicore()
    print("Detection Post processing finished")

    from evaluation.eval_detection import ANETdetection

    anet_detection = ANETdetection(
        ground_truth_filename="./evaluation/activity_net_1_3_new.json",
        prediction_filename=prediction_filename,
        subset='validation', verbose=False, check_status=False)
    anet_detection.evaluate()

    mAP_at_tIoU = [f'mAP@{t:.2f} {mAP*100:.3f}' for t, mAP in zip(anet_detection.tiou_thresholds, anet_detection.mAP)]
    results = f'Detection: average-mAP {anet_detection.average_mAP*100:.3f} {" ".join(mAP_at_tIoU)}'
    print(results)