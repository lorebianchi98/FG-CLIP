import argparse
import json
from PIL import Image
import pickle
from tqdm import tqdm  
from copy import deepcopy
import os
import math

from src.model import CrossAttentionModule
import torch
import clip      
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

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

def extract_scores_from_features(data, n_hardnegatives, batch_size, cross_attention=None, device='cuda'):
    images = {imm['id']: imm for imm in data['images']}
    categories = {cat['id']: cat for cat in data['categories']}
    
    len_vocabulary = n_hardnegatives + 1
    
    outputs = []
    
    # batch initialization
    batch_index = 0
    batch_images = None
    batch_vocabulary_ids = []
    batch_vocabularies = None
    batch_vocabulary_offsets = [0]
    batch_anns = []
    
    for ann in tqdm(data['annotations']):
        vocabulary, vocabulary_id, cats = create_vocabulary(ann, categories)
        
        batch_images = torch.cat((batch_images, ann['features'].unsqueeze(0)), dim=0) if batch_images is not None else ann['features'].unsqueeze(0)
        vocabulary_features = torch.stack([cat['features'] for cat in cats])
        # batch_vocabularies = torch.cat((batch_vocabularies, vocabulary_features.unsqueeze(0)), dim=0) if batch_vocabularies is not None else vocabulary_features.unsqueeze(0)
        batch_vocabularies = torch.cat((batch_vocabularies, vocabulary_features), dim=0) if batch_vocabularies is not None else vocabulary_features
        vocabulary_id = vocabulary_id[:len_vocabulary]
        
        batch_vocabulary_ids.append(vocabulary_id) 
        batch_vocabulary_offsets.append(batch_vocabulary_offsets[-1] + len(vocabulary))
        
        ann['bbox'][2] += ann['bbox'][0]
        ann['bbox'][3] += ann['bbox'][1]
        batch_anns.append(ann)
        batch_index += 1
        # if we are at the end of the batch, we make the inference
        if batch_index == batch_size or ann == data['annotations'][-1]:
            image_features = batch_images.to(device)
            text_features = batch_vocabularies.to(device)
            # retrieving the results
            for i in range(batch_images.shape[0]):
                # calculating cosine similarity
                txt_feats = text_features[batch_vocabulary_offsets[i]:batch_vocabulary_offsets[i] + len(batch_vocabulary_ids[i])]
                img_feats = image_features[i].expand(txt_feats.shape)
                with torch.no_grad():
                    if not cross_attention:
                        total_scores = F.cosine_similarity(img_feats, txt_feats, dim=1).tolist()
                    else:
                        total_scores = cross_attention(img_feats, txt_feats).tolist()
                outputs.append({
                    'labels': batch_vocabulary_ids[i][total_scores.index(max(total_scores))],
                    'boxes': batch_anns[i]['bbox'],
                    'scores': max(total_scores),
                    'category_id': batch_vocabulary_ids[i][0],
                    'vocabulary': batch_vocabulary_ids[i],
                    'image_filepath': images[batch_anns[i]['image_id']]['file_name'],
                    'total_scores': total_scores
                })
        
            # cleaning batch variables
            batch_index = 0
            batch_images = None
            batch_vocabulary_ids = []
            batch_vocabulary_offsets = [0]
            batch_vocabularies = None
            batch_anns = []
            
    
    # making one list of pred per category
    outputs = adjust_format(outputs)
    
    return outputs

def extract_scores_from_images(data, model, preprocess, coco_path, n_hardnegatives, batch_size, scale_factor, cross_attention=None):
    images = {imm['id']: imm for imm in data['images']}
    categories = {cat['id']: cat for cat in data['categories']}
    
    len_vocabulary = n_hardnegatives + 1
    
    outputs = []
    
    # batch initialization
    batch_index = 0
    batch_images = None
    batch_texts = []
    batch_vocabulary_ids = []
    batch_vocabularies = []
    batch_vocabulary_offsets = [0]
    batch_anns = []
    
    for ann in tqdm(data['annotations']):
        vocabulary, vocabulary_id, _ = create_vocabulary(ann, categories)
        # check if a number of hardnegatives is setted to non-default values
        # if it is, the vocabulary is clipped and if it is too short, we skip that image
        # if len(vocabulary) < len_vocabulary:
        #     continue
        vocabulary = vocabulary[:len_vocabulary]
        vocabulary_id = vocabulary_id[:len_vocabulary]
        
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
        batch_index += 1
        # if we are at the end of the batch, we make the inference
        if batch_index == batch_size or ann == data['annotations'][-1]:
            texts = clip.tokenize(batch_texts).to(device)
            with torch.no_grad():
                image_features = model.encode_image(batch_images)
                text_features = model.encode_text(texts)
                # retrieving the results
                for i in range(batch_images.shape[0]):
                    # calculating cosine similarity
                    txt_feats = text_features[batch_vocabulary_offsets[i]:batch_vocabulary_offsets[i] + len(batch_vocabulary_ids[i])]
                    img_feats = image_features[i].expand(txt_feats.shape)
                    if not cross_attention:
                        total_scores = F.cosine_similarity(img_feats, txt_feats, dim=1).tolist()
                    else:
                        total_scores = cross_attention(img_feats, txt_feats).tolist()
                    outputs.append({
                        'labels': batch_vocabulary_ids[i][total_scores.index(max(total_scores))],
                        'boxes': batch_anns[i]['bbox'],
                        'scores': max(total_scores),
                        'category_id': batch_vocabulary_ids[i][0],
                        'vocabulary': batch_vocabulary_ids[i],
                        'image_filepath': images[batch_anns[i]['image_id']]['file_name'],
                        'total_scores': total_scores
                    })
            
            # cleaning batch variables
            batch_index = 0
            batch_images = None
            batch_texts = []
            batch_vocabulary_ids = []
            batch_vocabulary_offsets = [0]
            batch_vocabularies = []
            batch_anns = []
            
    
    # making one list of pred per category
    outputs = adjust_format(outputs)
    
    return outputs
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to process')
    parser.add_argument('--coco_path', type=str, default="../coco/", help='Dataset to process')
    parser.add_argument('--out', type=str, required=True, help='Out path')
    parser.add_argument('--n_hardnegatives', type=int, default=10, help="Number of hardnegatives in each vocabulary")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--scale_factor', type=float, default=1.0, help="Set the percental dimension of the crop of the bounding box to process with CLIP")
    parser.add_argument('--model', type=str, default="ViT-B/16", help="CLIP model to use")
    parser.add_argument('--cross_attention', type=str, default=None, help="If setted, it applies cross-attention instead of a simple dot product. The value of this parameter is the path to the weight of the cross attention layer")
    args = parser.parse_args()
    
    if os.path.splitext(args.dataset)[1] == '.json':
        with open(args.dataset, "r") as file:
            data = json.load(file)
        feats_precomputed = False
    elif os.path.splitext(args.dataset)[1] == '.pt':
        feats_precomputed = True
        data = torch.load(args.dataset)
    else:
        print("Invalid dataset format.")
        return
      
    coco_path = args.coco_path
    
    model, preprocess = clip.load(args.model, device=device)
    
    if args.cross_attention:
        cross_attention = CrossAttentionModule(512, 64, 256, 2, 0.1)
        cross_attention.load_state_dict(torch.load(args.cross_attention))
        cross_attention.to(device)
        cross_attention.eval()
    else:
        cross_attention = None
    
    if not feats_precomputed:
        outputs = extract_scores_from_images(data, model, preprocess, coco_path, args.n_hardnegatives, args.batch_size, args.scale_factor, cross_attention)
    elif feats_precomputed:
        outputs = extract_scores_from_features(data, args.n_hardnegatives, args.batch_size, cross_attention)
    
    for i in range(args.n_hardnegatives + 1):
        clipped_outputs = clip_output(deepcopy(outputs), i)
        path = os.path.join(f"{args.model.replace('/', '-')}-{args.scale_factor}", f"preds{i}", f"{args.out}.pkl") if not args.cross_attention else os.path.join(f"{args.cross_attention.split('/')[-1][:-4]}", f"preds{i}", f"{args.out}.pkl")
        print("Saving " + path)
        with open(os.path.join(path), 'wb') as fid:
            pickle.dump(clipped_outputs, fid)
            
if __name__ == "__main__":
    main()