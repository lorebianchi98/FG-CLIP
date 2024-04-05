import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--model', type=str, default="ViT-B/16", help="CLIP model to use") # ViT-B/16, ViT-L/14
    parser.add_argument('--gpu', type=int, default="0", help="GPU to use")
    parser.add_argument('--cross_attention', type=str, default=None, help="If setted, it applies cross-attention instead of a simple dot product. The value of this parameter is the path to the weight of the cross attention layer")
    
    args = parser.parse_args()
    
    datasets = ['1_attributes', '2_attributes', '3_attributes', 'shuffle_negatives', 'color', 'material', 'transparency', 'pattern']
    scale_factors = [1.0]
    
    for scale_factor in scale_factors:
        model_path = f"{args.model.replace('/', '-')}-{scale_factor}" if not args.cross_attention else args.cross_attention.split('/')[-1][:-4]
        # create model and preds dir
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        for i in range(11):
            pred_path = os.path.join(model_path, f"preds{i}")
            # create model and preds dir
            if not os.path.exists(pred_path):
                os.makedirs(pred_path)
                
        
        for dataset in datasets:
            command = f"CUDA_VISIBLE_DEVICES={args.gpu} python inference_on_benchmark.py --dataset gt/{dataset}.json --out {dataset} --scale_factor {scale_factor} --model {args.model} --batch_size {args.batch_size} --cross_attention {args.cross_attention}"
            print(command)
            os.system(command)
        
    
    

if __name__ == "__main__":
    main()