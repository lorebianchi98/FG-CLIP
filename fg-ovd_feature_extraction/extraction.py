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

def create_vocabulary(ann, categories):
    vocabulary_id = [ann['category_id']] + ann['neg_category_ids']
    vocabulary = [categories[id]['name'] for id in vocabulary_id]
    cats = [categories[id] for id in vocabulary_id]
    return vocabulary, vocabulary_id, cats

def adjust_format(output):
    organized_preds = {}
    new_output = []
    for pred in output:
        if pred['category_id'] in organized_preds:
            organized_preds[pred['category_id']].append(pred)
        else:
            organized_preds[pred['category_id']] = [pred]

    for category_id, preds in organized_preds.items():
        new_pred = {
            'labels': [],
            'boxes': [],
            'scores': [],
            'category_id': category_id,
            'image_filepath': '',
            'vocabulary': [],
            'total_scores': []
        }

        for pred in preds:
            new_pred['image_filepath'] = pred['image_filepath']
            new_pred['vocabulary'] = pred['vocabulary']
            for k, v in pred.items():
                if k in ['category_id', 'image_filepath']:
                    continue
                new_pred[k].append(v)

        new_output.append(new_pred)

    return new_output

def clip_output(outputs, n_hardnegatives):
    new_outputs = []
    len_vocabulary = n_hardnegatives + 1

    for pred in outputs:
        assert len(pred['labels']) == len(pred['boxes']) == len(pred['scores']) == len(pred['total_scores']), "Incongruent dimensions"
        if len(pred['vocabulary']) < len_vocabulary:
            continue
        new_pred = deepcopy(pred)
        new_pred['vocabulary'] = new_pred['vocabulary'][:len_vocabulary]

        new_labels = []
        new_scores = []
        new_total_scores = []
        for total_score in new_pred['total_scores']:
            total_score = total_score[:len_vocabulary]
            new_scores.append(max(total_score))
            new_labels.append(total_score.index(max(total_score)) + pred['category_id'])
            new_total_scores.append(total_score)

        new_pred['labels'] = new_labels
        new_pred['scores'] = new_scores
        new_pred['total_scores'] = new_total_scores
        new_outputs.append(new_pred)

    return new_outputs

def extract_features(data, coco_path, batch_size, n_hardnegatives, model, preprocess, scale_factor):
    images = {imm['id']: imm for imm in data['images']}
    categories = {cat['id']: cat for cat in data['categories']}

    len_vocabulary = n_hardnegatives + 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # batch initialization
    batch_index = 0
    batch_images = None
    batch_texts = []
    batch_vocabulary_ids = []
    batch_vocabularies = []
    batch_vocabulary_offsets = [0]
    batch_anns = []
    batch_cats = []

    for ann in tqdm(data['annotations']):
        vocabulary, vocabulary_id, cats = create_vocabulary(ann, categories)
        vocabulary = vocabulary[:len_vocabulary]
        vocabulary_id = vocabulary_id[:len_vocabulary]
        cats = cats[:len_vocabulary]

        image_filepath = coco_path + images[ann['image_id']]['file_name']
        imm = Image.open(image_filepath)
        # cropping the bounding box
        bbox = ann['bbox'].copy()
        new_w = bbox[2] * scale_factor
        new_h = bbox[3] * scale_factor

        # Calculate the displacement from the center
        dx = (new_w - bbox[2]) // 2
        dy = (new_h - bbox[3]) // 2

        # Update the bounding box
        bbox[0] -= dx  # Update x1
        bbox[1] -= dy  # Update y1
        bbox[2] = new_w + bbox[0] # Update width
        bbox[3] = new_h + bbox[1] # Update height
        limits = [0, 0, imm.size[0], imm.size[1]]
        for ind in range(4):
            bbox[ind] = bbox[ind] if (ind < 2 and bbox[ind] >= 0) or (ind >= 2 and bbox[ind] <= limits[ind]) else limits[ind]
            bbox[ind] = math.floor(bbox[ind]) if ind < 2 else math.ceil(bbox[ind])

        imm = imm.crop(bbox)

        # adding images and captions to the batch
        imm_preprocessed = preprocess(imm).unsqueeze(0).to(device)
        if batch_images is None:
            batch_images = imm_preprocessed
        else:
            batch_images = torch.cat((batch_images, imm_preprocessed), dim=0)
        batch_texts += vocabulary
        batch_vocabulary_ids.append(vocabulary_id)
        batch_vocabularies.append(vocabulary)
        batch_vocabulary_offsets.append(batch_vocabulary_offsets[-1] + len(vocabulary))
        ann['bbox'][2] += ann['bbox'][0]
        ann['bbox'][3] += ann['bbox'][1]
        batch_anns.append(ann)
        batch_cats.append(cats)
        batch_index += 1
        # if we are at the end of the batch, we make the inference
        if batch_index == batch_size or ann == data['annotations'][-1]:
            texts = clip.tokenize(batch_texts).to(device)
            with torch.no_grad():
                image_features = model.encode_image(batch_images)
                text_features = model.encode_text(texts)
                # retrieving the results
                for i in range(batch_images.shape[0]):
                    # saving crop and vocabulary feature
                    batch_anns[i]['features'] = image_features[i]
                    txt_feats = text_features[batch_vocabulary_offsets[i]:batch_vocabulary_offsets[i] + len(batch_vocabulary_ids[i])]
                    for cat, feats in zip(batch_cats[i], txt_feats):
                        cat['features'] = feats

            # cleaning batch variables
            batch_index = 0
            batch_images = None
            batch_texts = []
            batch_vocabulary_ids = []
            batch_vocabulary_offsets = [0]
            batch_vocabularies = []
            batch_anns = []
            batch_cats = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='Dataset to process')
    parser.add_argument('--coco_path', type=str, default="../coco/", help='Dataset to process')
    parser.add_argument('--out_dir', type=str, required=True, help='Out path')
    parser.add_argument('--n_hardnegatives', type=int, default=10, help="Number of hardnegatives in each vocabulary")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--scale_factor', type=float, default=1.0, help="Set the percental dimension of the crop of the bounding box to process with CLIP")
    parser.add_argument('--model', type=str, default="ViT-B/16", help="CLIP model to use")

    args = parser.parse_args()

    benchmarks = ['1_attributes', '2_attributes', '3_attributes', 'shuffle_negatives', 'color', 'material', 'transparency', 'pattern']
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model, device=device)
    
    for bench in benchmarks:
        with open(os.path.join(args.dataset_dir, f"{bench}.json"), "r") as file:
            data = json.load(file)
        extract_features(data, args.coco_path, args.batch_size, args.n_hardnegatives, model, preprocess, args.scale_factor)
        torch.save(data, os.path.join(args.out_dir, f"{bench}.pt"))
        print(f"Saved {args.out_dir}/{bench}.pt")
        



if __name__ == "__main__":
    main()