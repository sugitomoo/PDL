import os
import torch
import h5py
import numpy as np
import json
from typing import Any, Union
from pathlib import Path
from os import PathLike
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
PathLike = Union[str, Path]

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ref https://github.com/TatsuyaShirakawa/KTS/blob/master/cpd_nonlin.pyx 
def calc_scatters(K,device):
    """Calculate scatter matrix: scatters[i,j] = {scatter of the sequence with
    starting frame i and ending frame j}
    """
    n = K.shape[0]
    K = K.to(device)

    K1 = torch.cumsum(torch.cat([torch.tensor([0],device=device),torch.diag(K)]),0)
    K2 = torch.zeros((n + 1, n + 1),device=device)
    K2[1:, 1:] = torch.cumsum(torch.cumsum(K, 0), 1)

    diagK2 = torch.diag(K2)

    i = torch.arange(n,device=device).reshape((-1, 1))
    j = torch.arange(n,device=device).reshape((1, -1))
    scatters = (
            K1[1:].reshape((1, -1)) - K1[:-1].reshape((-1, 1)) -
            (diagK2[1:].reshape((1, -1)) + diagK2[:-1].reshape((-1, 1)) -
             K2[1:, :-1].T - K2[:-1, 1:]) /
            ((j - i + 1).type(torch.float32) + (j == i - 1).type(torch.float32))
    )
    scatters[j < i] = 0

    return scatters

 
def cpd_nonlin(K, ncp, lmin=1, lmax=100000, backtrack=True, verbose=True,
               out_scatters=None,device='cpu'):
    """Change point detection with dynamic programming

    :param K: Square kernel matrix
    :param ncp: Number of change points to detect (ncp >= 0)
    :param lmin: Minimal length of a segment
    :param lmax: Maximal length of a segment
    :param backtrack: If False - only evaluate objective scores (to save memory)
    :param verbose: If true, print verbose message
    :param out_scatters: Output scatters
    :return: Tuple (cps, obj_vals)
        - cps - detected array of change points: mean is thought to be constant
            on [ cps[i], cps[i+1] )
        - obj_vals - values of the objective function for 0..m changepoints
    """

    m = int(ncp)  # prevent numpy.int64

    n, n1 = K.shape
    assert n == n1, 'Kernel matrix awaited.'
    assert (m + 1) * lmin <= n <= (m + 1) * lmax, f'show {(m + 1) * lmin} <= {n} <= {(m + 1) * lmax}'
    assert 1 <= lmin <= lmax

    if isinstance(K, np.ndarray):
        K_tensor = torch.from_numpy(K).float().to(device)
    else:
        K_tensor = K.float().to(device)
        
    J = calc_scatters(K_tensor, device)
    
    if out_scatters is not None:
        out_scatters[0] = J

    I = 1e101 * torch.ones((m + 1, n + 1),device=device)
    I[0, lmin:lmax] = J[0, lmin - 1:lmax - 1]

    if backtrack:
        p = torch.zeros((m + 1, n + 1), dtype=torch.int64,device=device)
    else:
        p = torch.zeros((1, 1), dtype=torch.int64,device=device)

    for k in range(1, m + 1):
        for l in range((k + 1) * lmin, n + 1):
            tmin = max(k * lmin, l - lmax)
            tmax = l - lmin + 1
            c = J[tmin:tmax, l - 1].reshape(-1) + \
                I[k - 1, tmin:tmax].reshape(-1)
            I[k, l] = torch.min(c)
            if backtrack:
                p[k, l] = torch.argmin(c) + tmin

    # Collect change points
    cps = torch.zeros(m, dtype=torch.int64,device=device)

    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k - 1] = p[k, cur]
            cur = cps[k - 1]

    scores = I[:, n]
    scores[scores > 1e32] = np.inf
    return cps, scores

def get_Mr_HiSum_video_id(youtube_id):
    df = pd.read_csv('../datasets/Mr_HiSum/metadata.csv')
    video_id = df.loc[df['youtube_id'] == youtube_id, 'video_id']
    return video_id.values[0]

def main(args):
    diversity_data = {}

    with open(f"../datasets/{args.dataset}/videos.json", "r") as file:
        videos = json.load(file)
         
    for video_name in videos:
        print(video_name) 
        video_dir = f"../datasets/{args.dataset}/videos/{video_name}"
        
        if args.dataset == "TVSum" or args.dataset == "SumMe":
            caption_feature_path = os.path.join(video_dir, "all_captions.npy") 
            with open(f"../datasets/{args.dataset}/comparison_{args.dataset}.json", "r") as f:
                video_name2number = json.load(f)
            video_number = video_name2number[video_name]
        
        elif args.dataset == "Mr_HiSum":
            caption_feature_path = f"../datasets/Mr_HiSum/videos/{video_name}/captions.npy"
            video_number = get_Mr_HiSum_video_id(video_name)
            
        video_information = h5py.File(f"../datasets/{args.dataset}/{args.dataset.lower()}.h5", "r")
        change_points = video_information[video_number]['change_points'][...].astype(np.int32) 

        feature_matrix = np.load(caption_feature_path)
        feature_tensor = torch.from_numpy(feature_matrix).float().to(device)
        kernel_matrix = torch.mm(feature_tensor,feature_tensor.T)
        
        num_change_points = len(change_points)-1
        min_length = 1
        max_length = 10000 
        
        cps , scores = cpd_nonlin(kernel_matrix, num_change_points, lmin=min_length, lmax=max_length, device=device)
        cps_indices = cps.cpu().numpy().astype(int)
        
        segments = []
        start_frame = 0
        frame_num = change_points[-1][1] + 1
        
        for cp in cps_indices:
            end_frame = cp - 1
            segments.append([start_frame,end_frame])
            start_frame = cp
            
        segments.append([start_frame,frame_num-1])
        segments_array = np.array(segments)
        np.save(os.path.join(video_dir, 'caption_segments.npy'), segments_array)

        scene_feature = np.array([feature_matrix[start:end+1].mean(axis=0) for start,end in segments_array])
        change = []
        
        for i in range(len(scene_feature)-1):
            similarity =  cosine_similarity([scene_feature[i]],[scene_feature[i+1]])[0][0]
            change.append(similarity)
            
        scene_change = [float(value) for value in change]
        sim_scene = float(np.mean(change))
        diversity_data[video_name] = {"s_change": scene_change, "sim_scene": sim_scene}
        
        output_path = f"../datasets/{args.dataset}/diversity.json"  
        print(diversity_data)
        
        with open(output_path, 'w') as f:
            json.dump(diversity_data, f, ensure_ascii=False, indent=4)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    
    main(args)