import argparse
import json
import os
import torch
import yaml

from itertools import product
from src.model import CrossAttentionModule, MLPs
from src.eval_util import get_mean_rank_sum
from src.metrics import get_image_and_text_tensor, i2t, t2i
from src.plots_util import bcolors

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test(model, retr_imm, retr_txt, datas):
    print("Evaluating PACCO Mean Rank Sum...")
    mean_rank_sum = get_mean_rank_sum(model, datas, 256, device)
    
    print("Evaluating COCO retrieval metrics...")
    r1i, r5i, r10i, _, _  = t2i(retr_imm.numpy(), retr_txt.numpy(), cross_attention=model)
    print("T2I")
    print(r1i, r5i, r10i)
    r1, r5, r10, _, _ = i2t(retr_imm.numpy(), retr_txt.numpy(), cross_attention=model)
    print("I2T")
    print(r1, r5, r10)
    return {
        'Mean Rank Sum': mean_rank_sum,
        'r1': r1,
        'r5': r5,
        'r10': r10,
        'r1i': r1i,
        'r5i': r5i,
        'r10i': r10i,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs_dir', type=str, default='configs', help='Configurations directory')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Checkpoints directory')
    parser.add_argument('--coco_test', type=str, default='./features/ViT-B-16/test.json', help='COCO test file')
    parser.add_argument('--pacco_test_dir', type=str, default='./fg-ovd_feature_extraction/gt', help='COCO test file')
    parser.add_argument('--results_file', type=str, default='eval/model_results.json', help='Output results file')
    parser.add_argument('--update_every_step', action='store_true', help='If setted, every step the results file will be overwritten')
    args = parser.parse_args()
    
    complete_results = {}
    if args.update_every_step:
        if os.path.exists(args.results_file):
            with open(args.results_file, 'r') as f:
                complete_results = json.load(f)
    
    # loading datasets
    retr_imm, retr_txt = get_image_and_text_tensor(args.coco_test)
    datas = {}
    for filename in os.listdir(args.pacco_test_dir):
        datas[filename.split('.')[0]] = torch.load(os.path.join(args.pacco_test_dir, filename), map_location=device)
    
    # calculating Vanilla CLIP results
    complete_results['CLIP B/16'] = test(None, retr_imm, retr_txt, datas)
    if args.update_every_step:
        with open(args.results_file, 'w') as f:
            json.dump(complete_results, f)
            
    
    
    train_config_dir = os.path.join(args.configs_dir, 'train')
    model_config_dir = os.path.join(args.configs_dir, 'model')
    train_configs = os.listdir(train_config_dir)
    model_configs = os.listdir(model_config_dir)
    print(f'Configuration to evaluate:\n{list(product(train_configs, model_configs))}')
    for train_config, model_config in list(product(train_configs, model_configs)):
        if not train_config.endswith('.yaml') or not model_config.endswith('.yaml'):
            continue
        
        model_name = train_config.split('.')[0] + '_' + model_config.split('.')[0]
        print(model_name)
        train_config_path = os.path.join(train_config_dir, train_config)
        model_config_path = os.path.join(model_config_dir, model_config)
        checkpoint_path = os.path.join(args.checkpoints_dir, f"{model_name}.pth")
        print(f"{bcolors.BOLD}{bcolors.UNDERLINE}{bcolors.GREEN}{model_name}{bcolors.ENDC}")
        
        config = {}
        with open(train_config_path, 'r') as config_file:
            config['train'] = yaml.safe_load(config_file)
        with open(model_config_path, 'r') as config_file:
            config['model'] = yaml.safe_load(config_file)
        print(f"Configuration loaded!\n{json.dumps(config, indent=2)}")
        # model loading
        if 'num_attention_layers' in config['model']:
            model = CrossAttentionModule.from_config(config['model'])
        else:
            model = MLPs.from_config(config['model'])
        model.load_state_dict(torch.load(checkpoint_path, device))
        model.to(device)
        complete_results[model_name] = test(model, retr_imm, retr_txt, datas)
        
        if args.update_every_step:
            with open(args.results_file, 'w') as f:
                json.dump(complete_results, f)
    with open(args.results_file, 'w') as f:
        json.dump(complete_results, f)
        
if __name__ == '__main__':
    main()