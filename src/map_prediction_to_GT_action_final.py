import sys  
sys.path.insert(0, '../src')
from util.eval_utils import standardize_dact
from multiprocess import Pool# use multiprocessing to speed up evaluation!

from pathlib import Path
import json
import os
import warnings
import numpy as np
import pandas as pd

DATASET = sys.argv[1]
TEST_SET = 'val' if DATASET == 'SGD' else 'test'
GT_DIR = f'../datasets/{DATASET}/action_prediction_gt_labels_{TEST_SET}_only'
PREDICTIONS_DIR = f'../outputs/{DATASET}/'
FILTER="*"

predictions = {}
domains = set()
models = set()
prompts = set()
num_data_per_domain = dict()
for pth in Path(PREDICTIONS_DIR).glob(FILTER):
    for config_pth in pth.glob('config*.json'):
        try:
            with config_pth.open() as cf:
                config = json.load(cf)
        except Exception as e:
            print(f'Error while reading {config_pth}')
            print(e)
            continue
        print(config_pth)
        for seed in config['seed']:
            for pred_pth in config_pth.parent.glob(f'DM_prediction_S{seed}.json'):
                dataset = config['dataset']
                if '_trajectories.json' in config['traj_path']:
                    domain = config['traj_path'].rsplit('/', 1)[-1].split('_trajectories.json')[0]
                else:
                    domain = config['traj_path'].rsplit('/', 1)[-1].split('.json')[0]
                model = config['model']
                prompt = (config['prompt_style'].replace('_', '-'), config['num_shot'], config['use_mask_prompt'])
                temp = 0.0 if 'temperature' not in config else config['temperature']
                sampling = 'multi' if 'sampling' not in config else config['sampling']
                
                domains.add(domain)
                models.add(model)
                prompts.add(prompt)
                key = (domain, model, prompt, temp, sampling, seed, pred_pth)
                try:
                    with pred_pth.open() as f:
                        predictions[key] = json.load(f)
                    #print(pred_pth)
                    if domain not in num_data_per_domain:
                        num_data_per_domain[domain] = 1
                    else:
                        num_data_per_domain[domain] += 1
                except Exception as e:
                    print(f'Error while reading {pred_pth}')
                    print(e)
                    break
print(f"num data: {len(predictions)}")
num_data = list(num_data_per_domain.values())
num_consistent = [curr_num == num_data[0] for curr_num in num_data_per_domain.values()]
if not all(num_consistent):
    print('Error: number of data is inconsistent across domains')
    print(num_data_per_domain)
    #assert False

gt_labels, gt_mapped_labels, gt_mapped_multi_labels = {}, {}, {}
# GT files
count = 0
for domain in domains:
    matching = list(Path(GT_DIR).glob(f'{domain}_labels.json'))
    if len(matching) == 0:
        warnings.warn(f'{domain}: GT labels not found!')
    else:
        pth = matching[0]
        with pth.open('r') as f:
            gt_labels[domain] = json.load(f)
        count += 1
print(f"num GT labels loaded: {count}")

def map_prediction_into_file(args):
    pred_params, traj_pred = args
    domain, model, prompt_params, temp, sampling, seed, pred_pth = pred_params
    prompt_style, num_shot, use_mask_prompt = prompt_params
    is_multisampling = float(temp) > 0 and ('repeat' not in sampling)
    
    pred_dump_path = str(pred_pth).replace("_prediction_", "_mapped_prediction_").replace(".json",".npy")
    
    gt_processed_label_tuple = standardize_dact(gt_labels[domain], traj_pred, near_option_mapping=True, multi=is_multisampling)
    
    processed_label_for_save = np.array(gt_processed_label_tuple, dtype=object)
    print(f"dumping at {pred_dump_path}")
    np.save(pred_dump_path, processed_label_for_save, allow_pickle=True)

jobs = []
for pred_params, traj_pred in predictions.items():
    jobs.append((pred_params, traj_pred))

with Pool(min(60, len(jobs))) as p:
    results = [result for result in p.imap(map_prediction_into_file, jobs) if result is not None]
