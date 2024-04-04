import argparse
import json
from PIL import Image
import pickle
from tqdm import tqdm
from copy import deepcopy
import os
import math

import torch
import clip
import torch.nn.functional as F

def out2coco_format(data, imm_outs, ann_outs):
    to_copy = ['imm_feats', 'dense_out']
    for imm in data['images']:
        for x in to_copy:
            imm[x] = imm_outs[imm['id']][x]
            
    to_copy = ['ann_feats', 'dense_out']
    for ann in data['annotations']:
        for x in to_copy:
            ann[x] = ann_outs[ann['id']][x]
            
    return data

def reduce_data(data, n_elem):
    # images = {imm['id']: imm for imm in data['images']}
    # data['annotations'] = data['annotations'][:n_elem]
    # data['images'] = list(set([images[ann['image_id']] for ann in data['annotations']]))
    data['images'] = data['images'][:n_elem]
    images = {imm['id']: imm for imm in data['images']}
    data['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] in list(images.keys())]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=str, required=True, help='Annotations file path')
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--coco_path', type=str, default="../coco/", help='Dataset to process')
    parser.add_argument('--model', type=str, default="ViT-B/16", help="CLIP model to use") # ViT-B/16, ViT-L/14
    parser.add_argument('--out', type=str, default="out.pt", help="Out path")
    args = parser.parse_args()

    torch.manual_seed = 123

    with open(args.annotations, "r") as file:
        data = json.load(file)
    
    split = 'train2014' if 'train2014' in args.annotations else 'val2014'
    coco_path = os.path.join(args.coco_path, split)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model, device=device)

    imm_outs = {}
    # images features extraction:
    n_batches = (len(data['images']) + args.batch_size - 1) // args.batch_size
    for i in tqdm(range(n_batches)):
        low_idx = i * args.batch_size
        up_idx = low_idx + min(len(data['images']) - i, args.batch_size)
        # iterate over batch images
        batch_imms, batch_imm_ids = None, []
        for imm in data['images'][low_idx:up_idx]:
            batch_imm_ids.append(imm['id'])
            image_filepath = os.path.join(coco_path, imm['file_name'])
            imm_preprocessed = preprocess(Image.open(image_filepath)).unsqueeze(0).to(device)
            if batch_imms is None:
                batch_imms = imm_preprocessed
            else:
                batch_imms = torch.cat((batch_imms, imm_preprocessed), dim=0)
        # calculate the features
        with torch.no_grad():
            batch_imm_feats, batch_dense_out = model.encode_image(batch_imms, dense=True)
        # move to cpu
        batch_imm_feats = batch_imm_feats.to('cpu')
        batch_dense_out = batch_dense_out.to('cpu')
        for imm_id, imm_feats, dense_out in zip(batch_imm_ids, batch_imm_feats, batch_dense_out):
            imm_outs[imm_id] = {
                'imm_feats': imm_feats,
                'dense_out': dense_out
            }

    ann_outs = {}
    # text extraction
    n_batches = (len(data['annotations']) + args.batch_size - 1) // args.batch_size
    for i in tqdm(range(n_batches)):
        low_idx = i * args.batch_size
        up_idx = low_idx + min(len(data['annotations']) - i, args.batch_size)
        # iterate over batch annotations
        batch_anns, batch_ann_ids = [], []
        for ann in data['annotations'][low_idx:up_idx]:
            batch_ann_ids.append(ann['id'])
            batch_anns.append(ann['caption'])
        
        # calculate the features
        batch_anns = clip.tokenize(batch_anns).to(device)
        with torch.no_grad():
            batch_ann_feats, batch_dense_out = model.encode_text(batch_anns, dense=True)
        # move to cpu
        batch_ann_feats = batch_ann_feats.to('cpu')
        batch_dense_out = batch_dense_out.to('cpu')
        for ann_id, ann_feats, dense_out in zip(batch_ann_ids, batch_ann_feats, batch_dense_out):
            ann_outs[ann_id] = {
                'ann_feats': ann_feats,
                'dense_out': dense_out
            }

    new_data = out2coco_format(data, imm_outs, ann_outs)
    torch.save(new_data, args.out)


if __name__ == "__main__":
    main()