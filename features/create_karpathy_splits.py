import argparse
from copy import deepcopy
from itertools import chain
import json
import os
import torch
from tqdm import tqdm

def get_karpathy_splits(data, splits):
    return {imm['cocoid']: imm['sentids'][:5] for imm in data['images'] if imm['split'] in splits}


def get_split_featured(split, datas):
    new_data = {}
    new_data['annotations'] = []
    new_data['images'] = []
    
    anns = list(chain(*split.values()))
    
    for imm in tqdm(datas['images']):
        if imm['id'] in split.keys():
            new_data['images'].append(imm)
            
    
    for ann in tqdm(datas['annotations']):
        if ann['id'] in anns:
            new_data['annotations'].append(ann)
            
    return new_data




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_splits', type=str, default="../coco/dataset_coco.json", help="JSON where the splits are defined")
    parser.add_argument('--src_features_dir', type=str, default="ViT-B-16/old", help="Dir where the features extracted from CLIP are kept")
    parser.add_argument('--out_dir', type=str, default="ViT-B-16", help="Out directory")
    args = parser.parse_args()
    
    
    with open(args.target_splits, 'r') as f:
        data = json.load(f)

    # train = get_karpathy_splits(data, ['train', 'restval'])
    # val = get_karpathy_splits(data, ['val'])
    test = get_karpathy_splits(data, ['test'])
    test1k = dict(list(test.items())[:1000])

    # # loading data with extracted features
    print("loading features")
    train_data = torch.load(os.path.join(args.src_features_dir, "train.json"))
    val_data = torch.load(os.path.join(args.src_features_dir, "val.json"))
    
    datas = {}
    datas['images'] = train_data['images'] + val_data['images']
    datas['annotations'] = train_data['annotations'] + val_data['annotations']
    
    # imms2anns = [train, val, test, test1k]
    imms2anns = [test1k]
    names = ['train', 'val', 'test']
    names = ['test1k']
    
    for imm2ann, name in zip(imms2anns, names):
        print(f"Preparing {name} split. N. images: {len(imm2ann.keys())} N. annotations: {len(imm2ann.values())}")
        split = get_split_featured(imm2ann, datas)
        torch.save(split, os.path.join(args.out_dir, f'{name}.json'))
        print(f"Done! N. images: {len(split['images'])} N. annotations: {len(split['annotations'])}")
    
    

if __name__ == '__main__':
    main()