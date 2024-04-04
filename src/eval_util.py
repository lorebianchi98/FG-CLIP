import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import tqdm 

from copy import deepcopy
from fgovd_evaluation.inference_on_benchmark import extract_scores_from_features
from src.metrics import i2t, t2i
from src.model import CrossAttentionModule, MLPs
from src.plots_util import bcolors

def dict2pandas(data, current_mapping):
    data = pd.json_normalize(data)
    data.columns = pd.MultiIndex.from_tuples(data.columns.str.split('.').tolist())
    data = data.transpose().reset_index().rename(columns={
        'level_0': 'detector',
        'level_1': 'benchmark',
        'level_2': 'num_negatives',
        'level_3': 'metric',
        0: 'value',
    }).replace({
        'detector': current_mapping,
        'benchmark': {
            'shuffle_negatives': 'Trivial',
            '3_attributes': 'Easy',
            '2_attributes': 'Medium',
            '1_attributes': 'Hard',
            'color': 'Color',
            'material': 'Material',
            'pattern': 'Pattern',
            'transparency': 'Transparency',
        },
        'metric': {
            'map': 'mAP',
            'median': 'Median Rank',
            'medium': 'Mean Rank',
            'positions_array': 'Rank',
            'predictions_scores': 'Score',
        }
    }).convert_dtypes()

    return data

def get_ranks(outputs, limit=None):
    """
    Return medium and median rank, and the avg. scores of positive caption, best negative caption and negative captions
    """
    position_arrays = {i: [] for i in range(1,11)}

    def get_median_rank(position_array):
        return sorted(position_array)[(len(position_array) // 2)]
            
    def get_medium_rank(position_array):
        return sum(elem for elem in position_array) / len(position_array)
    
    sum_pos = 0
    sum_best_neg = 0
    sum_negs = 0
    max_lens = 0 if limit is None else limit
    for preds in outputs:
        for scores in preds['total_scores']:
            for n_neg in range(1, len(scores)):
                if (limit is None and n_neg == len(scores) - 1) or (limit is not None and n_neg == limit):
                    max_lens = limit if limit is not None else max(max_lens, len(scores) - 1) # saving max len of negatives
                    sum_pos += scores[0]
                    sum_best_neg += max(scores[1:])
                    sum_negs += sum(scores[1:]) / len(scores[1:])
                
                new_scores = scores[:n_neg + 1]
                # rank = sorted(new_scores, reverse=True).index(new_scores[0]) + 1
                rank = sum(1 for conf in new_scores if conf >= new_scores[0])
                position_arrays[n_neg].append(rank)
    
    
    results = {str(n_neg): {
        'medium': get_medium_rank(position_arrays[n_neg]) if len(position_arrays[n_neg]) > 1 else -1,
        'median': get_median_rank(position_arrays[n_neg]) if len(position_arrays[n_neg]) > 1 else -1,
        }
        for n_neg in range(1, 11)
    }
    
    return results, \
           sum_pos / len(position_arrays[max_lens]), \
           sum_best_neg / len(position_arrays[max_lens]), \
           sum_negs / len(position_arrays[max_lens])
           
           

def do_fgovd_eval(model, model_name, gt_dir, n_neg=10, batch_size=256, verbose=False):
    device = next(model.parameters()).device
    total_results = {}
    benchmarks = {
            'shuffle_negatives': 'Trivial',
            '3_attributes': 'Easy',
            '2_attributes': 'Medium',
            '1_attributes': 'Hard',
            'color': 'Color',
            'material': 'Material',
            'pattern': 'Pattern',
            'transparency': 'Transparency',
    }
    
    # benchmarks = ['1_attributes', '2_attributes', '3_attributes', 'shuffle_negatives', 'color', 'material', 'transparency', 'pattern']    
    total_results[model_name] = {} 
    stats = {
        'bench': list(benchmarks.values()),
        'Avg. pos. score': [],
        'Avg. best neg. score': [],
        'Avg. negs. score': [],
        }
    print("Calculating models ranks in FG-OVD Benchmarks...")
    for bench in benchmarks.keys():
        data = torch.load(os.path.join(gt_dir, bench + ".pt"), map_location=device)
        outputs = extract_scores_from_features(deepcopy(data), n_neg, batch_size, model, device)
        # using custom rank function simplified
        if bench == 'transparency':
            limit = 2
        elif bench == 'pattern':
            limit = 7
        else:
            limit = 10
        ranks, avg_pos, avg_best_neg, avg_negs = get_ranks(deepcopy(outputs), limit)
        stats['Avg. pos. score'].append(avg_pos)
        stats['Avg. best neg. score'].append(avg_best_neg)
        stats['Avg. negs. score'].append(avg_negs)
        if verbose:
            print(f"Avg confidence positive: {avg_pos} --- Avg confidence best negative: {avg_best_neg} --- Avg confidence negatives: {avg_negs}")
        total_results[model_name][bench] = {}
        for i in range(1, n_neg + 1):
            # skipping high number of negatives transparencies and patterns
            if i > 2 and bench == 'transparency':
                continue
            if i > 7 and bench == 'pattern':
                continue 
            total_results[model_name][bench][str(i)] = {}
            # clipped_outputs = clip_output(deepcopy(outputs), i)
            # rank = CustomMetrics()
            # rank.update(deepcopy(data), clipped_outputs, n_neg = i)
            # mean, median = rank.get_medium_rank(), rank.get_median_rank()
            
            total_results[model_name][bench][str(i)]['medium'] = ranks[str(i)]['medium']
            total_results[model_name][bench][str(i)]['median'] = ranks[str(i)]['median']
            if verbose:
                print(f"{bench} {i} negatives --- Mean Rank = {ranks[str(i)]['medium']}, Median Rank = {ranks[str(i)]['median']}")
    
    print()
    
    df = pd.DataFrame(stats)
    df = df.round(2)
    print(df)
    print()
    
    
    return total_results

def plot_rank(data, plot_detectors=["CLIP B/16", "Current Method"]):
    benchmarks = ['Hard', 'Medium', 'Easy', 'Trivial']
    attributes = ['Color', 'Material', 'Pattern', 'Transparency']
    sns.set_theme(style="whitegrid", context="talk", font='serif', font_scale=0.8, rc={'text.usetex': False, 'text.latex.preamble': r'\usepackage{underscore}'})

    plot_data = data.query('metric == "Mean Rank" & detector in @plot_detectors').explode('value', ignore_index=True) #.assign(value=lambda x: 1 / x.value)
    palette = matplotlib.colormaps.get_cmap('rainbow')
    palette = [palette(i) for i in np.linspace(0, 1, len(plot_detectors))]
    markers = ['D', 'o', 'X', 's', 'P', 'd']

    grid = sns.catplot(data=plot_data, kind='point', height=2.4, aspect=1.3, palette=palette, col_wrap=4,
        x='num_negatives', y='value', hue='detector', col='benchmark', hue_order=plot_detectors, col_order=benchmarks+attributes,
        markers=markers, # marker='o,
        legend='full', errorbar=None, estimator='median', dodge=0.25, markersize=5, linewidth=1, facet_kws=dict(despine=False),
    )

    # move legend to last axes
    bbox = grid.axes[7].get_position()
    sns.move_legend(grid, title='Detector', loc='lower right', bbox_to_anchor=(1.16*bbox.x1, 0.9*bbox.y0), 
        handlelength=1.0, handletextpad=0.5, columnspacing=0.3, frameon=True, ncol=2)

    # set texts
    grid.set_titles(col_template="{col_name}")
    grid.set(xlabel=None, ylabel='Rank')
    grid.fig.text(0.5, 0.01, r'Number of negative captions $N$', ha='center', va='center')

    # add major and minor ticks with autolocator
    for i, ax in enumerate(grid.axes.flat):
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.grid(which='major', axis='x', visible=True)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['top'].set_visible(True)

    grid.axes.flat[0].invert_yaxis()

    grid.fig.tight_layout()
    # fig_filename = f'mean-rank-vs-negatives-CLIPvsOWL.pdf'
    # grid.fig.savefig(fig_filename, bbox_inches='tight')
    
    

def do_eval(model, model_name, images=None, texts=None):
    gt_dir = 'fg-ovd_feature_extraction/gt'
    results = do_fgovd_eval(model, model_name, gt_dir)
    with open('baseline.json', 'r') as f:
        baseline = json.load(f)
    results.update(baseline)
    current_mapping = {
        'ViT-B-16': "CLIP B/16",
        model_name: model_name
    }
    data = dict2pandas(results, current_mapping)
    plot_rank(data, list(current_mapping.values()))
    
    if images is not None and texts is not None:
        print()
        print(f"{bcolors.BOLD}Retrieval metrics on COCO{bcolors.ENDC}")
        print("(r1i, r5i, r10i, medri, meanri)")
        print(t2i(images.numpy(), texts.numpy(), cross_attention=model))
        print("(r1, r5, r10, medr, meanr)")
        print(i2t(images.numpy(), texts.numpy(), cross_attention=model))


def get_mean_rank_sum(model, datas, batch_size=64, device='cuda', verbose=False):
    n_neg = 10
    
    mean_rank_sum = 0
    iterable = datas.items() if not verbose else tqdm(datas.items())
    for bench, data in iterable:
        outputs = extract_scores_from_features(data, n_neg, batch_size, model, device)
        if bench == 'transparency':
            limit = 2
        elif bench == 'pattern':
            limit = 7
        else:
            limit = 10
        ranks, _, _, _ = get_ranks(outputs, limit)
        mean_rank_sum += ranks[str(limit)]['medium']
        
    return mean_rank_sum
def main():
    import yaml 
    
    val_dir = './fg-ovd_feature_extraction/val_sets'
    benchs = ['shuffle_negatives', '3_attributes', '2_attributes', '1_attributes', 'color', 'material', 'pattern', 'transparency']
    datas = {}
    
    with open('./configs/triplet_1atts_rand.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    for bench in benchs:
        datas[bench] = torch.load(f"{val_dir}/{bench}.pt", map_location='cuda')
    
    model = CrossAttentionModule.from_config(config['model'])
    model.to('cuda')
    print(get_mean_rank_sum(None, datas))

if __name__ == '__main__':
    main()