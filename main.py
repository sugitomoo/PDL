import os
import torch
import numpy as np
import math
from typing import Any, Union
from pathlib import Path
from os import PathLike
from utils import *
from model import *
from loss import *
from dataset import *
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def text_matching(args,caption_path,summary_path, D_score):
    """
    Performs text matching between individual captions and the text summary using the Siamese-Sentence BERT architectures.
    """  
    model = SiameseSentenceBERT().to(device)
    criterion =  ImprovedMarginRankingLoss(args.margin) 
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
        
    num_epochs = 100
    caption_lines = read_caption(caption_path)
    summary_lines = read_summary_json(summary_path)

    loss_values = []
    last_loss = None
    relative_change_threshold = 0.01
    
    lam = (1 - D_score) * math.exp(1 - D_score)
    if D_score >= args.delta:
        lam = 0 
        
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        similarities = model(caption_lines, summary_lines)
        normalized_similarities = (similarities + 1) / 2
        loss = criterion(normalized_similarities)
        
        if args.alpha_sparsity:
            sparsity = sparsity_loss()
            loss += sparsity(normalized_similarities,args.sparsity_target) * args.alpha
        
        elif (args.PDL) and (lam!=0):
            sparsity = sparsity_loss()
            loss += sparsity(normalized_similarities,args.sparsity_target) * lam
            
        loss.backward()
        optimizer.step()
       
        total_loss += loss.item()
        loss_values.append(total_loss)

        if last_loss is not None:
            relative_change = (last_loss - total_loss) / ((last_loss)+1e-5) #avoid zero division
            if relative_change < relative_change_threshold:
                    break
        last_loss = total_loss
        
    model.eval()
    with torch.no_grad():
        new_similarities = model(caption_lines, summary_lines)
    new_similarities = (new_similarities + 1 ) /2 
    score = new_similarities.squeeze().cpu().numpy()
    return score
    
def main(args):
    set_random_seed(args.seed)
    
    with open(f"./datasets/{args.dataset}/videos.json","r") as f:
        videos = json.load(f)
    
    video_dataset = VideoSumDataset(videos, args.dataset)
    total_kendall_tau = 0
    total_spearman_rho = 0

    for i in range(len(video_dataset)):
        video_name, gtscore, change_points, n_frames, n_frame_per_seg, picks, user_summary, user_score, caption_path, summary_path, D_score = video_dataset[i]
        print(video_name)
        video_dir = f"./datasets/{args.dataset}/videos/{video_name}"
        
        pred = text_matching(args, caption_path,summary_path, D_score)
        pred_summary_kp,frame_scores = get_keyshot_summ(pred,change_points,n_frames,n_frame_per_seg,picks,proportion=float(0.15),seg_score_mode=str('mean'),method='knapsack')
        
        if args.dataset == 'SumMe' or args.dataset == 'TVSum':
            kendall_tau, spearman_rho = calc_kendall_spearman(frame_scores, user_score)
            print(f"Kendall's tau: {kendall_tau}")
            print(f"Spearman's rho: {spearman_rho}")
            total_kendall_tau+= kendall_tau
            total_spearman_rho += spearman_rho   
            path = f"{args.margin}_{args.lr}_{kendall_tau:.4f}_{spearman_rho:.4f}"
        elif args.dataset == "Mr_HiSum":
            path = f"{args.margin}_{args.lr}"

        frame_score_path = os.path.join(video_dir,f"frame_score_{path}.npy")
        selected_index_path = os.path.join(video_dir,f"summary_{path}.npy") 
        if args.save:
            np.save(frame_score_path, pred)
            np.save(selected_index_path,pred_summary_kp)  
        
        if args.dataset == "Mr_HiSum":
            continue       

        
    avg_kendall_tau = total_kendall_tau / len(video_dataset)
    avg_spearman_rho = total_spearman_rho / len(video_dataset)
    print(f"Average Kendall's tau: {avg_kendall_tau}")
    print(f"Average Spearman's rho: {avg_spearman_rho}") 
    
    if args.alpha_sparsity:
        save_result_name = f"alpha_{args.alpha}_{args.lr}_{args.margin}.txt"
    elif args.PDL:
        save_result_name = f"PDL_{args.lr}_{args.margin}.txt"
    else:
        save_result_name = f"{args.lr}.txt"
        
    os.makedirs(f'./datasets/{args.dataset}/results/',exist_ok=True)
    with open(f'./datasets/{args.dataset}/results/{save_result_name}', 'a') as file:
        file.write(f"{args.margin}\t{avg_kendall_tau}\t{avg_spearman_rho}\n")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--margin',type=float,default=0.11)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--sparsity',action='store_true')
    parser.add_argument('--alpha_sparsity', action="store_true")
    parser.add_argument('--alpha',type=float)
    parser.add_argument('--PDL',action='store_true')
    parser.add_argument('--sparsity_target',type=float,default=0.3)
    parser.add_argument('--delta',type=float,default=0.35)
    parser.add_argument('--save', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    main(args)