import json
from openai import OpenAI
import os
import argparse

def load(filename):
    return json.load(open(filename, 'r'))

def read_caption_to_list(caption_path):
    with open(caption_path,'r') as file:
        lines = file.readlines()
        lines = [line.strip().rstrip('.') for line in lines if line.strip()]
    return lines

def get_metadata(args,video_name):
    meta_path = f"../datasets/{args.dataset}/metadata.json"
    with open(meta_path,'r') as file:
        metadata = json.load(file)
        
    for item in metadata:
        if (args.dataset == "SumMe") and (item['original title'] == video_name):
            return item['Substituted title']
        elif (args.dataset == "TVSum") and (item['video_id'] == video_name):
            return item['title'], item['genre'], item['query']
        elif (args.dataset == "Mr_HiSum") and (item['video_id'] == video_name):
            return item['Substituted title']
    
def main(args):
    TOKEN = load('settings.json')['token']
    client = OpenAI(api_key=TOKEN)
    
    with open(f"../datasets/{args.dataset}/videos.json", "r") as file:
        videos = json.load(file) 
        
    for video_name in videos:
        print(video_name)
        video_dir = f"../datasets/{args.dataset}/videos/{video_name}" 
        caption_path = os.path.join(video_dir,"captions.txt")
        captions = "\n".join(read_caption_to_list(caption_path))

        general_prompt = open(f"general_summary_{args.dataset}.txt","r").read()
        personalize_prompt = open(f"personalize.txt","r").read() 
        
        prompt = general_prompt.replace('[CAPTIONS]',captions) 
        
        if args.dataset == "SumMe" or args.dataset == "Mr_HiSum":
            title = get_metadata(args, video_name)
            prompt = prompt.replace('[TITLE]',title)
        
        elif args.dataset == "TVSum":
            title, genre, query = get_metadata(args,video_name)
            prompt = prompt.replace('[TITLE]',title).replace('[GENRE]',genre).replace('[QUERY]',query)
            
        output_file = os.path.join(video_dir,"summary.json")
        
        if args.personalize:
            prompt = personalize_prompt.replace('[CAPTIONS]',captions)
            output_file = os.path.join(video_dir,"personalize.json")
        print(prompt)
        
        response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "You are an expert in summarization."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        )
        
        json_string = response.choices[0].message.content
        data = json.loads(json_string)
        print(response)
        
        with open(output_file, 'w') as f:
            json.dump(data, f,indent=4)  

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument("--personalize", action="store_true")
    args = parser.parse_args()
    
    main(args)
