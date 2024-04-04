import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--model', type=str, default="ViT-B/16", help="CLIP model to use") # ViT-B/16, ViT-L/14
    parser.add_argument('--gpu', type=int, default="0", help="GPU to use")
    args = parser.parse_args()
    
    out_dir = args.model.replace('/', '-')
    
    # create model and preds dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    splits = ['train', 'val']
    for split in splits:
        command = f"CUDA_VISIBLE_DEVICES={args.gpu} python coco_feature_extractor.py --annotations ../coco/annotations/captions_{split}2014.json --batch_size {args.batch_size} --model {args.model} --out {os.path.join(out_dir, split + '_tmp.json')}"
        print(command)
        os.system(command)
        
    
    

if __name__ == "__main__":
    main()