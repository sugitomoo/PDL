# Language-Guided Self-Supervised Video Summarization Using Text Semantic Matching Considering the Diversity of the Video (ACMMM Asia 2024)
### [Paper](https://arxiv.org/abs/2405.08890)
The official repository of our paper "**Language-Guided Self-Supervised Video Summarization Using Text Semantic Matching Considering the Diversity of the Video**".

## Model Overview
<p align="center">
<img src="./fig/PDL_model.pdf" alt="model" width="80%">
</p>


## Requirements
```bash
conda create -n PDL python=3.8.13
conda activate PDL
pip install torch==2.1.2 torchvision==0.16.2 transformers==4.37.2 sentence-transformers
pip install pandas h5py pillow av openai opencv-python 
```

## Dataset
### SumMe and TVSum Datasets
Please download the original videos and place them under each `original_videos` directory.

The structured h5 files with annotations for the SumMe and TVSum datasets are available from [SSPVS](https://github.com/HopLee6/SSPVS-PyTorch), named `summe.h5` and `tvsum.h5`.

Each dataset has its own `metadata.json` file, which includes information obtained from the original YouTube videos, such as the substituted title.
The `comparison_SumMe.json` and `comparison_TVSum.json` contain the correspondence between video indices and video names in the respective datasets. These are available from [google drive link](https://drive.google.com/drive/folders/1bF8v9KlPo3gT6Rhnx4H4lDDIVJKUBGcw)



### Mr.HiSum Dataset
Please download `mr_hisum.h5` and `metadata.csv` from [the Mr.HiSum dataset](https://github.com/MRHiSum/MR.HiSum).

The generated personalized videos are available from [google drive link](https://drive.google.com/drive/folders/1-6JubvrzfstoObLzmvfknkKmSucOtPHU?usp=sharing).

The files should be organized in the following structure:
   ```
    ├── datasets
        └── SumMe
            ├── original_videos
            ├── comparison_SumMe.json
            ├── metadata.json
            ├── summe.h5
            ├── videos.json
        └── TVSum
            ├── original_videos
            ├── comparison_TVSum.json
            ├── metadata.json
            ├── tvsum.h5
            ├── videos.json
        └── Mr_HiSum
            ├── original_videos
            ├── mr_hisum.h5
            ├── metadata.json
            ├── metadata.csv
            ├── videos.json
   ```

## Preprocessing
Please place your OpenAI token in the `settings.json` file.

We generate captions from individual downsampled video frames. To follow existing works, we genearate the captions with `frame_interval=15` for calculating frame scores and `frame_interval=1` for segment detection. 
For the Mr.HiSum dataset, the frames are downsampled to 1 fps by adjusting the `frame_interval` following existing works.

```bash
python create_captions.py --dataset=${dataset} \
    --frame_interval=${frame_interval} 
```

We calculate the diversity of the video using the generated captions.
```bash
python calculate_diversity.py --dataset=${dataset} 
```

The generated captions are synthesized into text summaries.
When we perform personalized video summarization, replace the query in [USER QUERY] with `personalize.txt` and add `--personalize` as an argument.

```bash
python create_summaries.py --dataset=${dataset} (--personalize)
```

## Language-Guided Video Summarization
Please run the following command, setting the margin and the learning rate for each dataset

```bash
python main.py --dataset=${dataset} --margin=${margin} --lr=${lr} --PDL  
```


## Citation
If you find our paper or code useful in your work, please **[★star]** this repo and **[cite]** the following publication:

BibTeX:
```bibtex
@article{sugihara2024language,  
  title={Language-Guided Self-Supervised Video Summarization Using Text Semantic Matching Considering the Diversity of the Video},  
  author={Sugihara, Tomoya and Masuda, Shuntaro and Xiao, Ling and Yamasaki, Toshihiko},  
  journal={arXiv preprint arXiv:2405.08890},  
  year={2024}  
}  
```
