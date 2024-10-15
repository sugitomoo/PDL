import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import av
import json
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

processor = AutoProcessor.from_pretrained("microsoft/git-large-textcaps")
captioning_model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-textcaps").to(device)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)


def generate_captions_from_video(video_path,args):
    """
    Generate captions from individual video frames using a pre-trained model (GIT).
    """
    container = av.open(video_path)
    captions = []  

    for i, frame in enumerate(container.decode(video=0)):
        if i % args.frame_interval == 0:  
            img = frame.to_image()  
            
            prompt = "a photo of"
            pixel_values = processor(images=img, text=prompt, return_tensors="pt").pixel_values.to(device)
            generated_ids = captioning_model.generate(pixel_values=pixel_values, max_length=20)
            generated_caption = processor.batch_decode(generated_ids,skip_special_tokens=True)[0]            

            # omit the prompt
            clean_caption = generated_caption.replace(prompt, '').strip()
            captions.append(clean_caption)
            
    return captions

# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
def sentence_features(file_path):
    """
    Extract caption features using a pre-trained Sentence-BERT.
    """
    features = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line:
                features.append(model.encode(line))
    return np.array(features)  

def main(args):
    with open(f"../datasets/{args.dataset}/videos.json", "r") as file:
        videos = json.load(file)
    
    for video_name in videos:
        print(video_name)
        video_dir = f"../datasets/{args.dataset}/videos/{video_name}"
        original_video_dir = f"../datasets/{args.dataset}/original_videos"
        
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        
        video_path = os.path.join(original_video_dir, f"{video_name}.mp4")
        captions = generate_captions_from_video(video_path, args)
        
        if args.frame_interval == 1:
            caption_path = os.path.join(video_dir, "all_captions.txt")
            caption_feature_path = os.path.join(video_dir, "all_captions.npy")
        else:
            caption_path = os.path.join(video_dir, "captions.txt")
            caption_feature_path = os.path.join(video_dir, "captions.npy")
         
        with open(caption_path, 'w') as file:
            file.write('\n'.join(captions))
            
        caption_features = sentence_features(caption_path)
        np.save(caption_feature_path, caption_features)     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--frame_interval", type=int)
    args = parser.parse_args()
    
    main(args)
