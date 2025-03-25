import os
import h5py
import numpy as np
import json
from typing import Any, Union
from pathlib import Path
from os import PathLike
from utils import *
PathLike = Union[str, Path]

class VideoSumDataset(object):
    def __init__(self, videos, dataset):
        """
        Initializes the VideoSumDataset class.

        Args:
            videos (list): List of video names.
            dataset (str): Name of the dataset (e.g., "TVSum", "SumMe", "Mr_HiSum").
        """
        self.videos = videos
        self.dataset = dataset
        self.data_root = f'./datasets/{dataset}'
        self.video_dict = h5py.File(f'{self.data_root}/{dataset.lower()}.h5', 'r')
        self.comparison_path = f"{self.data_root}/comparison_{dataset}.json"
         
    def __len__(self):
        """
        Returns the number of videos in the dataset.
        """
        return len(self.videos)

    def __getitem__(self, index):
        """
        Returns the information of each video.
        """
        video_name = self.videos[index]
        
        if self.dataset == "Mr_HiSum":
            video_number = get_Mr_HiSum_video_id(video_name)
            
        else:
            with open(self.comparison_path, 'r') as f:
                video_name2number = json.load(f)
            video_number = video_name2number[video_name]

        video_file = self.video_dict[video_number]
        gtscore = video_file['gtscore'][...].astype(np.float32) # [T]
        change_points = video_file['change_points'][...].astype(np.int32) # [S, 2], S: number of segments, each row stores indices of a segment          

        if self.dataset == "Mr_HiSum":
            with open(f"{self.data_root}/metadata.json",'r') as file:
                metadata = json.load(file)
            for item in metadata:
                if item['video_id'] == video_name:
                    frame_interval = item['frame_interval']
                    n_frames = item['original_frame_number']
            n_frame_per_seg = ((change_points[:,1] - change_points[:,0] + 1) * frame_interval)
            picks = np.array(list(range(0, n_frames, frame_interval)))
            user_summary = False
            user_score = False
        else:
            n_frames = video_file['n_frames'][...].astype(np.int32) 
            n_frame_per_seg = video_file['n_frame_per_seg'][...].astype(np.int32) 
            picks = video_file['picks'][...].astype(np.int32) 
            user_summary = video_file['user_summary'][...].astype(np.float32)
            user_score = video_file['user_scores'][...].astype(np.float32) 
 
        caption_path = os.path.join(self.data_root, "videos", video_name, "captions.txt") 
        summary_path = os.path.join(self.data_root, "videos", video_name, "summary.json") 

        with open(f"{self.data_root}/diversity.json", "r") as f:
            diversity_information = json.load(f)   
        sim_scene = diversity_information[video_name]["sim_scene"]
        D_score = 1 - sim_scene
        
        return video_name, gtscore, change_points, n_frames, n_frame_per_seg, picks, user_summary, user_score, caption_path, summary_path, D_score
