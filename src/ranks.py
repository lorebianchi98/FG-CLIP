from tqdm import tqdm

import torch
from torch import IntTensor, Tensor
from torchvision.ops import batched_nms

from src.utilities import get_image_ground_truth, transform_predslist_to_dict, get_image_preds, calculate_iou
from src.utilities import read_json, load_object


# Code borrowed from https://github.com/lorebianchi98/FG-OVD/blob/main/evaluation/ranks.py
class CustomMetrics():
    intersected_predictions = []
    position_array = []
    
    def __init__(self) -> None:
        intersected_predictions = []
        position_array = []
        pass
    
    def update(self, targets, preds, nms=True, iou=0.5, verbose=False, one_inference_at_time=False, n_neg=10):
        """
        target: Detection dataset in standard COCO format
        preds: List of detections from a dataset where each one has the following fields: 'scores', 'boxes' (in the format xyxy), 'labels', 'total_socres', 'image_id'
        """
        
        targets['annotations'] = [ann for ann in targets['annotations'] if len(ann['neg_category_ids']) >= n_neg]
        
        if one_inference_at_time:
            return self.__update_one_inference_at_time__(targets, preds, iou=iou, verbose=verbose)
        
        self.intersected_predictions = []
        self.position_array = []
        
        # transforming list to dict
        preds = transform_predslist_to_dict(preds)
        
        # initializing counters
        n_images = 0
        count_no_intersections = 0
        
        # iterating over images
        if verbose:
            targets['images'] = tqdm(targets['images'])
        for imm in targets['images']:
            target = get_image_ground_truth(targets, imm['id'])
            if imm['file_name'] in preds:
                imm_preds = [get_image_preds([pred_per_cat]) for pred_per_cat in preds[imm['file_name']]]
            else:
                continue
            n_images += 1
            # iterating over the predictions per category
            for imm_preds_per_cat in imm_preds:
                # appliyng NMS to preds if nms is True
                if nms:
                    to_keep = batched_nms(imm_preds_per_cat['boxes'], 
                                        imm_preds_per_cat['scores'], 
                                        torch.IntTensor([0] * len(imm_preds_per_cat['boxes'])),
                                        iou)
                    imm_preds_per_cat['boxes'], imm_preds_per_cat['scores'], imm_preds_per_cat['labels'], imm_preds_per_cat['total_scores'] = imm_preds_per_cat['boxes'][to_keep], \
                                                                                                            imm_preds_per_cat['scores'][to_keep], \
                                                                                                            imm_preds_per_cat['labels'][to_keep], \
                                                                                                            imm_preds_per_cat['total_scores'][to_keep]
                # iterating over targets
                for box, label, id in zip(target['boxes'], target['labels'], target['annotation_id']):
                    if label != imm_preds_per_cat['category_id']:
                        continue
                    # applying NMS class agnostics to the preds concatenated with target, with targets score = 1
                    to_remove = batched_nms(torch.cat((imm_preds_per_cat['boxes'], box.unsqueeze(0)), dim=0), 
                                            torch.cat((imm_preds_per_cat['scores'], IntTensor([1])), dim=0), 
                                            torch.IntTensor([0] * (len(imm_preds_per_cat['boxes']) + 1)),
                                            iou)
                    # check which is the deleted box with higher confidences
                    deleted_elements = sorted(list(set(range(len(imm_preds_per_cat['scores']))) - set(to_remove.tolist()[1:])))
                    if len(deleted_elements) > 0:
                        # iou of deleted element needs to be over IoU with the GT
                        assert calculate_iou(imm_preds_per_cat['boxes'][deleted_elements[0]].tolist(), box) >= iou, "iou of deleted element needs to be over IoU with the GT"
                        
                        # appending to the list of predicted confidence the element
                        self.intersected_predictions.append({
                            'annotation_id': id,
                            'total_scores': imm_preds_per_cat['total_scores'][deleted_elements[0]].tolist()
                        })
                    else:
                        count_no_intersections += 1
                        self.intersected_predictions.append({
                            'annotation_id': id,
                            'total_scores': imm_preds_per_cat['total_scores'].shape[1] * [0]
                        })
                    # we store the rank of the prediction in the position array congruently
                    # we see the rank as the number of confidence in the list with higher or equal value - 1
                    last_pred = self.intersected_predictions[-1]
                    rank = sum(1 for conf in last_pred['total_scores'] if conf >= last_pred['total_scores'][0]) - 1
                    self.position_array.append(rank + 1)
        
        
        assert len(self.intersected_predictions) == len(self.position_array), "Incongruent list dimensions"
        if verbose:
            print("Number of images: %d" % n_images)
            print("Number of no intersection with GT: %d" % count_no_intersections)
        return self.intersected_predictions, self.position_array
    
    def get_median_rank(self):
        return sorted(self.position_array)[(len(self.position_array) // 2) + 1]
            
    def get_medium_rank(self):
        return sum(elem for elem in self.position_array) / len(self.position_array)
    
    # PRIVATE METHODS
    def __update_one_inference_at_time__(self, targets, preds, nms=True, iou=0.5, verbose=False, one_inference_at_time=False):
        """
        target: Detection dataset in standard COCO format
        preds: List of detections from a dataset where each one has the following fields: 'scores', 'boxes' (in the format xyxy), 'labels', 'total_socres', 'image_id'
        """
        
        self.intersected_predictions = []
        self.position_array = []
        
        # transforming list to dict
        preds = transform_predslist_to_dict(preds)
        
        # initializing counters
        n_images = 0
        count_no_intersections = 0
        
        def assign_max_scores(scores, ious, labels, iou_thresh):
            label_offset = int(min(labels))
            total_scores = [0] * (int(max(labels) - min(labels)) + 1)

            for score, iou, label in zip(scores, ious, labels):
                if iou < iou_thresh:
                    continue
                if total_scores[int(label - label_offset)] < score:
                    total_scores[int(label - label_offset)] = float(score)
            
            return total_scores

        # iterating over images
        if verbose:
            targets['images'] = tqdm(targets['images'])
        for imm in targets['images']:
            target = get_image_ground_truth(targets, imm['id'])
            if imm['file_name'] in preds:
                imm_preds = [get_image_preds([pred_per_cat], include_total_scores=False) for pred_per_cat in preds[imm['file_name']]]
            else:
                continue
            n_images += 1
            # iterating over the predictions per category
            for imm_preds_per_cat in imm_preds:
                for box, label, id in zip(target['boxes'], target['labels'], target['annotation_id']):
                    ious = [float(calculate_iou(box, pred_box)) for pred_box in imm_preds_per_cat['boxes']]
                    total_scores = assign_max_scores(imm_preds_per_cat['scores'], ious, imm_preds_per_cat['labels'], iou)
                    self.intersected_predictions.append({
                        'annotation_id': id,
                        'total_scores': total_scores
                    })
                    # we store the rank of the prediction in the position array congruently
                    # we see the rank as the number of confidence in the list with higher or equal value - 1
                    last_pred = self.intersected_predictions[-1]
                    rank = sum(1 for conf in last_pred['total_scores'] if conf >= last_pred['total_scores'][0]) - 1
                    self.position_array.append(rank + 1)
                
        return self.intersected_predictions, self.position_array

def clip_preds(data, n):
    
    for pred_group in data:
        new_vocabulary = pred_group['vocabulary'][:n + 1]
        keep = [i for i in range(len(pred_group['labels'])) if pred_group['labels'][i] in new_vocabulary][:50]
        pred_group['boxes'] = [pred_group['boxes'][i] for i in keep]
        pred_group['labels'] = [pred_group['labels'][i] for i in keep]
        pred_group['scores'] = [pred_group['scores'][i] for i in keep]
        
    return data

import json, pickle

def main():
    bench = 'transparency'
    # model = 'ViT-B-16-1.0'
    model = '64heads'
    n_neg = 1
    print("Defining CustomMetrics...")
    custom_metrics = CustomMetrics()
    path = f"../gt/{bench}.json"
    with open(path, "r") as f:
        gt = json.load(f)
    print(path)
    path = f"../{model}/preds{n_neg}/{bench}.pkl"
    with open(path, "rb") as f:
        preds = pickle.load(f)
    print(path)
    custom_metrics.update(gt, preds, n_neg=2)
    print("Medium: %s" % custom_metrics.get_medium_rank())
    print("Median: %s" % custom_metrics.get_median_rank())
    
    
if __name__ == "__main__":
    main()
            
        
