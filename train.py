import argparse
import json
import os
import torch
import yaml

from src.model import CrossAttentionModule, MLPs
from src.dataset import COCO2CLIPDataset, PACCO2CLIPDataset
from src.train_util import do_train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config', type=str, required=True, help='Training configuration')
    parser.add_argument('--model_config', type=str, required=True, help='Model configuration')
    parser.add_argument('--data_path', type=str, default="features/ViT-B-16", help='Directory where the dataset is stored')
    parser.add_argument('--out_dir', type=str, default="checkpoints", help='Out directory where the model checkpoint will be saved')
    parser.add_argument('--save_loss', action='store_true', help='If setted, loss will be saved at args.out_dir/loss')
    parser.add_argument('--skip_warmup', action='store_true', help='If setted, no warmup will be done')
    
    args = parser.parse_args()
    
    if args.train_config.split('.')[-1] != 'yaml' or args.model_config.split('.')[-1] != 'yaml':
        raise ValueError("Config must be a YAML file")
    
    warmup = not args.skip_warmup
    
    train_type = args.train_config.split("/")[-1].split(".")[-2]
    model_type = args.model_config.split("/")[-1].split(".")[-2]
    model_name = f"{train_type}_{model_type}"
    out_path = os.path.join(args.out_dir, f'{model_name}.pth')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = {}
    with open(args.train_config, 'r') as config_file:
        config['train'] = yaml.safe_load(config_file)
    with open(args.model_config, 'r') as config_file:
        config['model'] = yaml.safe_load(config_file)
    # if we are in the fg-ovd case, we have to load warmup weight of the model
    # otherwise no
    # remove these 2 lines if you want to load initial_weights outside fgovd
    if not config['train'].get('fgovd', False):
        config['model']['initial_weights'] = None 
    
    print(f"Configuration loaded!\n{model_name}\n{json.dumps(config, indent=2)}")
    # model loading
    if 'num_attention_layers' in config['model']:
        model = CrossAttentionModule.from_config(config['model'])
    else:
        model = MLPs.from_config(config['model'])
    model.to(device)
    # dataset loading
    fgovd = config['train'].get('fgovd', False)
    if not fgovd:
        train_dataset = COCO2CLIPDataset('./features/ViT-B-16/train.json')
        val_dataset = COCO2CLIPDataset('./features/ViT-B-16/val.json')
        additional_val_dataset = None
    else:
        train_dataset = PACCO2CLIPDataset('./fg-ovd_feature_extraction/training_sets/1_attributes.pt')
        val_dataset = PACCO2CLIPDataset('./fg-ovd_feature_extraction/val_sets/1_attributes.pt')
        additional_val_dataset = COCO2CLIPDataset('./features/ViT-B-16/val.json')
    
    if args.save_loss:
        loss_dir = os.path.join(args.out_dir, 'loss')
        if not os.path.exists(loss_dir):
            # If not, create it
            os.makedirs(loss_dir)
            print(f"Directory '{loss_dir}' created.")
        loss_path = os.path.join(loss_dir, f'{model_name}.jpg')
    else:
        loss_path = None
    
    results_dir = os.path.join(args.out_dir, 'results')
    if not os.path.exists(results_dir):
        # If not, create it
        os.makedirs(results_dir)
        print(f"Directory '{results_dir}' created.")
    results_path = os.path.join(results_dir, f"{model_name}.pt")
    
    model = do_train(model, train_dataset, val_dataset, config['train'], plot=False, loss_path=loss_path, additional_val_dataset=additional_val_dataset, results_path=results_path, warmup=warmup)
    torch.save(model.state_dict(), out_path)
    
    
if __name__ == '__main__':
    main()