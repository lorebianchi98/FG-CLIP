# FG-CLIP
[![arXiv](https://img.shields.io/badge/arXiv-2404.03539-b31b1b.svg)](https://arxiv.org/abs/2404.03539) 

Official repository of the paper **"Is CLIP the main roadblock for fine-grained open-world perception?"**.

This repository contains the code to perform training and evaluation of CLIP on the object crop of the [FG-OVD](https://github.com/lorebianchi98/FG-OVD) training sets and benchmarks. 

The `checkpoints` directory stores the parameters obtained from these trainings. To utilize these pre-trained CLIP projections which enhance CLIP's fine-grained understanding without repeating the training process, please refer to [Load weights](#load-weights). 
# Updates
- :fire: 09/2024: **"Is CLIP the main roadblock for fine-grained open-world perception?"** won the *Best Paper Award* at CBMI 2024!
# Installation

```bash
conda create --name clip python=3.9 -y
conda activate clip
git clone --recursive https://github.com/lorebianchi98/FG-CLIP.git
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
cd CLIP
python setup.py install
cd ..
pip install -r requirements.txt
```
**NOTE**: This project uses a custom version of CLIP because it allows us to extract all tokens from the visual and textual encoders, not just the CLS token.
If your goal is to extract only the CLS token (as done in the standard usage of this repo), you can install the official version of CLIP from the [official CLIP repository](https://github.com/openai/CLIP).


# Feature Extraction
To accelerate the training process, we utilize pre-extracted CLIP features within this repository. Please adhere to the following guidelines for feature extraction.
## COCO
To set up the required data for feature extraction, follow these steps:

1. Create a folder named "coco" using the following command:
    ```bash
    mkdir coco
    ```

2. Download the 2014 images and annotations from the [official COCO website](https://cocodataset.org/#download). Ensure that you place the downloaded files inside the newly created "coco" folder.

3. Run the following commands to start coco feature extraction:
    ```bash
    cd features
    python extract.py --gpu GPU_NUMBER --batch_size BATCH_SIZE --model MODEL
    # Example extraction command:
    # python extract.py --gpu 0 --batch_size 16 --model ViT-B/16
    ```
4. Download the [COCO Karpathy splits](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits). Run the following commands to move the pre-extracted feature in the Karpathy splits:
    ```bash
    python create_karpathy_splits.py --target_splits KARPATHY_SPLITS_DIR --src_features_dir COCO_FEATURES_DIR --out_dir OUT_DIR
    ```
## FG-OVD Benchmarks
1. To pre-extract features from the FG-OVD Benchmarks, download it from the [official FG-OVD repository](https://lorebianchi98.github.io/FG-OVD/).
2. Run the following commands:
    ```bash
    cd fg-ovd_feature_extraction
    # scale_factor: multiplier to the coordinates of the bounding boxes, the higher the value, the higher the context of the crop. Default = 1.0
    # model: CLIP configuration to use. Default = ViT-B/16
    python extraction --dataset_dir FG-OVD_DIR -coco_path COCO_PATH --out_dir OUT_DIR --batch_size BATCH_SIZE --scale_factor SCALE_FACTOR --model MODEL 
    ```

# Vanilla CLIP vs. FG-OVD Evaluation
Run the following commands:
```bash
cd fgovd_evaluation/
python main.py
```
This will create outputs in the format of [FG-OVD](https://github.com/lorebianchi98/FG-OVD). Use the script in the original [repo](https://github.com/lorebianchi98/FG-OVD/blob/main/evaluation/ranks.py) or refer to __get_ranks__ inside this [script](https://github.com/lorebianchi98/FG-OVD_CLIP_Evaluation/blob/main/src/eval_util.py).

# CLIP Fine-grained repurpose
## Training
To perform a training, run the following command:
```bash
CUDA_VISIBLE_DEVICES=GPU python train.py --train_config configs/train/TRAINING_CONFIG --model_config configs/model/MODEL_CONFIG
```

## Evaluation
This command will create a JSON with the results on both COCO and the FG-OVD dataset for each possible configuration:

```bash
CUDA_VISIBLE_DEVICES=GPU PYTHONPATH=. python plots/test.py --results_file OUT
```
## Load weights
To load a configuration with a set of saved weights, edit the corresponding yaml file in `configs/model` and set the field `initial_weights` to the path of the your checkpoints.
Then, in your python script, run the following commands:
```python
from src.model import CrossAttentionModule, MLPs
model = MLPs.from_config(MODEL_PATH) # CrossAttentionModule.from_config(MODEL_PATH)

# usage example with 2 as batch size
image_embeddings = torch.rand(2, 512) 
text_embeddings = torch.rand(2, 512)
similarities = model(image_embeddings, text_embeddings)
# in case you are using MLPs, you can also extract image and text embeddings repurposed
repurposed_image_embeddings, repurposed_text_embeddings = model(image_embeddings, text_embeddings, ret_embeds=True)
```


# Reference
If you found this code useful, please cite the following paper:
```
@inproceedings{bianchi2024clip,
  author       = {Lorenzo Bianchi and
                 Fabio Carrara and
                 Nicola Messina and
                 Fabrizio Falchi},
title        = {Is Clip the Main Roadblock for Fine-Grained Open-World Perception?},
booktitle    = {21st International Conference on Content-Based Multimedia Indexing,
                 {CBMI} 2024, Reykjavik, Iceland, September 18-20, 2024},
pages        = {1--8},
publisher    = {{IEEE}},
year         = {2024},
url          = {https://doi.org/10.1109/CBMI62980.2024.10859215},
doi          = {10.1109/CBMI62980.2024.10859215}
} 
 
```
