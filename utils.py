import os
import random
from pathlib import Path
from os import PathLike
from typing import Any, List, Dict
import re
import numpy as np
import torch
import json
import cv2
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import  spearmanr, kendalltau, rankdata

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_caption(caption_path):
    with open(caption_path,'r') as file:
        caption_lines = [line.strip() for line in file.readlines()]
    return caption_lines

def read_summary_json(summary_path):
    with open(summary_path,'r') as file:
        summary = json.load(file)
    last_summary = summary['Summaries'][-1]
    summary_lines = re.split(r'[.!?]',last_summary['Denser_Summary'])
    summary_lines = [line.strip() for line in summary_lines if line.strip()]
    return summary_lines

def read_personal_summary_json(summary_path):
    with open(summary_path,'r') as file:
        summary = json.load(file)
    summary_lines = re.split(r'[.!?]',summary['summary'])
    summary_lines = [line.strip() for line in summary_lines if line.strip()]
    return summary_lines

def get_Mr_HiSum_video_id(youtube_id):
    df = pd.read_csv('./datasets/Mr_HiSum/metadata.csv')
    video_id = df.loc[df['youtube_id'] == youtube_id, 'video_id']
    return video_id.values[0]

def check_inputs(values,weights,n_items,capacity):
    assert(isinstance(values,list))
    assert(isinstance(weights,list))
    assert(isinstance(n_items,int))
    assert(isinstance(capacity,int))
    assert(all(isinstance(val,int) or isinstance(val,float) for val in values))
    assert(all(isinstance(val,int) for val in weights))
    assert(all(val >= 0 for val in weights))
    assert(n_items > 0)
    assert(capacity > 0)

#ref https://github.com/HopLee6/SSPVS-PyTorch
def knapsack_dp(values,weights,n_items,capacity,return_all=False):
    check_inputs(values,weights,n_items,capacity)

    table = np.zeros((n_items+1,capacity+1),dtype=np.float32)
    keep = np.zeros((n_items+1,capacity+1),dtype=np.float32)

    for i in range(1,n_items+1):
        assert i <= len(weights), f"Index i out of range: {i}" 
        
        for w in range(0,capacity+1):
            wi = weights[i-1] 
            vi = values[i-1] 
            if (wi <= w) and (vi + table[i-1,w-wi] > table[i-1,w]):
                table[i,w] = vi + table[i-1,w-wi]
                keep[i,w] = 1
            else:
                table[i,w] = table[i-1,w]

    picks = []
    K = capacity

    for i in range(n_items,0,-1):
        if keep[i,K] == 1:
            picks.append(i)
            K -= weights[i-1]

    picks.sort()
    picks = [x-1 for x in picks] 
    if return_all:
        max_val = table[n_items,capacity]
        return picks,max_val
    return picks


#ref https://github.com/boheumd/A2Summ
def get_keyshot_summ(pred: np.ndarray,
                     cps: np.ndarray,
                     n_frames: int,
                     nfps: np.ndarray,
                     picks: np.ndarray,
                     proportion: float = 0.15,
                     seg_score_mode: str = 'mean',
                     method: str = 'knapsack'
                     ) -> np.ndarray:
    """
    Generate keyshot-based video summary i.e. a binary vector.

    :param pred: Predicted importance scores.
    :param cps: Change points, 2D matrix, each row contains a segment.
    :param n_frames: Original number of frames.
    :param nfps: Number of frames per segment.
    :param picks: Positions of subsampled frames in the original video.
    :param proportion: Max length of video summary compared to original length.
    :return: Generated keyshot-based summary.
    """
    picks = np.asarray(picks, dtype=np.int32)
    assert pred.shape == picks.shape, "pred:{} picks:{}".format(pred.shape, picks.shape)
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    for i in range(len(picks)):
        pos_lo = picks[i]
        pos_hi = picks[i + 1] if i + 1 < len(picks) else n_frames
        frame_scores[pos_lo:pos_hi] = pred[i]

    # Assign scores to video shots as the average of the frames.
    seg_scores = np.zeros(len(cps), dtype=np.int32)
    for seg_idx, (first, last) in enumerate(cps):
        scores = frame_scores[first:last + 1]
        if seg_score_mode == 'mean':
            seg_scores[seg_idx] = int(1000 * scores.mean())
        elif seg_score_mode == 'sum':
            seg_scores[seg_idx] = int(1000 * scores.sum())

    # Apply knapsack algorithm to find the best shots
    limits = int(round(n_frames * proportion))
    if method == 'knapsack':
        packed = knapsack_dp(seg_scores.tolist(), nfps.tolist(),int(len(cps)), limits)
    elif method == "rank":
        order = np.argsort(seg_scores)[::-1].tolist()
        packed = []
        total_len = 0
        for i in order:
            if total_len + nfps[i] < limits:
                packed.append(i)
                total_len += nfps[i]

    else:
        raise KeyError("Unknown method {}".format(method))

    summary = np.zeros(n_frames, dtype=bool)
    for seg_idx in packed:
        first, last = cps[seg_idx]
        summary[first:last + 1] = True
    return summary, frame_scores

def calc_kendall_spearman(machine_summary, user_summary):
    taus = []
    rhos = []

    if len(user_summary.shape) > 1:
        for annotator_summary in user_summary:
            rho = spearmanr(rankdata(-np.array(annotator_summary)), rankdata(-np.array(machine_summary)))[0]
            rhos.append(rho)
            tau = kendalltau(rankdata(-np.array(annotator_summary)), rankdata(-np.array(machine_summary)))[0]
            taus.append(tau)
    else:
        rho = spearmanr(rankdata(-np.array(user_summary)), rankdata(-np.array(machine_summary)))[0]
        rhos.append(rho)
        tau = kendalltau(rankdata(-np.array(user_summary)), rankdata(-np.array(machine_summary)))[0]
        taus.append(tau)

    average_kendall_tau = sum(taus) / len(taus)
    average_spearman_rho = sum(rhos) / len(rhos)

    return average_kendall_tau, average_spearman_rho 
