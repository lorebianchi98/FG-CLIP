from torch.utils.data import Dataset
import torch



class COCO2CLIPDataset(Dataset):
    def __init__(self, features_file):
        print("Loading dataset...")
        data = torch.load(features_file, map_location='cpu')
        print("Dataset loaded!")
        images = {imm['id']: imm for imm in data['images']}
        del data['images']
        self.data = {}
        for idx, ann in enumerate(data['annotations']):
            ann_id = ann['id']
            imm_id = ann['image_id']
            self.data[idx] = {}
            self.data[idx]['annotation'] = ann['ann_feats'] 
            self.data[idx]['image'] = images[imm_id]['imm_feats']
            self.data[idx]['image_id'] = imm_id
            self.data[idx]['annotation_id'] = ann_id
            
    def __getitem__(self, idx):
        annotation = self.data[idx]['annotation']
        image = self.data[idx]['image']
        metadata = {
            'annotation_id': self.data[idx]['annotation_id'],
            'image_id': self.data[idx]['image_id']
        }
        
        return {
            'annotation': annotation,
            'image': image,
            'metadata': metadata,
        }
    
    def __len__(self):
        return len(self.data)
    
class PACCO2CLIPDataset(Dataset):
    def __init__(self, features_file, n_capts=11):
        # TODO: currently we are discarding annotations which have less then n_capts captions (positive + negatives). We can modify the code to pad these annotation instead of discarding them and block the loss function score for the paddings 
        print("Loading dataset...")        
        data = torch.load(features_file, map_location='cpu')
        print("Dataset loaded!")
        self.data = {}
        data['annotations'] = [ann for ann in data['annotations'] if n_capts <= (len(ann['neg_category_ids']) + 1)]
        categories = {cat['id']: cat for cat in data['categories']}
        for idx, ann in enumerate(data['annotations']):
            # to stay consistent with COCO2CLIPDataset, in this case data[idx]['image'] refer to features of the annotation bbox
            # while data[idx]['annotation] refers to the categories features (shape [n_capts, 512]) 
            ann_id = ann['id']
            cat_ids = [ann['category_id']] + ann['neg_category_ids']
            self.data[idx] = {}
            self.data[idx]['image'] = ann['features'] 
            self.data[idx]['annotation'] = torch.stack([categories[id]['features'] for id in cat_ids], dim = 0) # n_neg x 512
            self.data[idx]['image_id'] = ann['image_id']
            self.data[idx]['annotation_id'] = ann_id
            
    def __getitem__(self, idx):
        annotation = self.data[idx]['annotation']
        image = self.data[idx]['image']
        metadata = {
            'annotation_id': self.data[idx]['annotation_id'],
            'image_id': self.data[idx]['image_id']
        }
        
        return {
            'annotation': annotation,
            'image': image,
            'metadata': metadata,
        }
    
    def __len__(self):
        return len(self.data)


def main():
    dataset = PACCO2CLIPDataset('../fg-ovd_feature_extraction/training_sets/1_attributes.pt')
if __name__ == '__main__':
    main()