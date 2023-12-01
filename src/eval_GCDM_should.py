import sys
sys.path.insert(0, '../src')
from pathlib import Path
import json
import os
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd

SHD_ALG_NAME = "SHDILP"
GRAPH_ALG_NAME = "CSILP"
FILTER="*"

DATASET = sys.argv[1]
INF_GRAPH_DIR = f'../graphs/{DATASET}_final/{GRAPH_ALG_NAME}'
SHD_DIR = f'../graphs/{DATASET}_final/{SHD_ALG_NAME}'
#
PDATASET = 'FINAL_SGD_single_multi_sample' if DATASET == 'SGD' else 'MWF_combined'
PREDICTIONS_DIR = f'../outputs/{PDATASET}/'

def read_config(config_pth):
    try:
        with config_pth.open() as cf:
            config = json.load(cf)
    except Exception as e:
        print(f'Error while reading {config_pth}')
        print(e)
    #dataset = config['dataset']
    if '_trajectories.json' in config['traj_path']:
        config['domain'] = config['traj_path'].rsplit('/', 1)[-1].split('_trajectories.json')[0]
    else:
        config['domain'] = config['traj_path'].rsplit('/', 1)[-1].split('.json')[0]
    prompt = (config['prompt_style'], config['num_shot'], config['use_mask_prompt'])
    temp = 0.0 if 'temperature' not in config else config['temperature']
    sampling = 'multi' if 'sampling' not in config else config['sampling']
    
    key = [config['domain'], config['model'], prompt, temp, sampling]
    return config, key

domains = set()
predictions, mapped_predictions = {}, {}
predictions_by_domain, label_tuple_by_domain = {}, {}
print(f'searching {len(list(Path(PREDICTIONS_DIR).glob(FILTER)))} directories @ {PREDICTIONS_DIR}')
for pth in Path(PREDICTIONS_DIR).glob(FILTER):
    for config_pth in pth.glob('config*.json'):
        config, base_key = read_config(config_pth)
        domains.add(config['domain'])
        for seed in config['seed']:
            key = tuple(base_key + [seed])
            for mapped_file_pth in config_pth.parent.glob(f'DM_mapped_prediction_S{seed}.npy'):
                if key in mapped_predictions:
                    assert False, f'duplicated: {key}, {mapped_file_pth}'
                try:
                    mapped_predictions[key] = np.load(mapped_file_pth, allow_pickle=True)
                    #print(f'loading @ {mapped_file_pth}')
                except Exception as e:
                    print(f'Error while reading {mapped_file_pth}', e)
        #print(mapped_predictions.keys())
        label_tuple_by_domain[config['domain']] = (mapped_predictions[key][-2], mapped_predictions[key][-1])
assert len(mapped_predictions) % len(domains) == 0, "Number of predictions per domain is not consistent!"
print(f"Loaded {len(mapped_predictions)} files for {len(domains)} domains")

from util.graph_utils import get_graph_sop
def load_graphs(domains_list, root_dir, dataset, graph_alg_name, label_tuple_by_domain, is_should, filter="*"):
    load_count = 0
    num_graph_per_domain = []
    graphs = {}
    print(f'Loading {root_dir}/{dataset}_"domain"/*{graph_alg_name}{filter}.npy')
    for domain in domains_list:
        print(f'loading @ {root_dir}/{dataset}_{domain}*/*{graph_alg_name}*.npy')
        all_acts, all_statuses = label_tuple_by_domain[domain]
        graph_algo_dict = {}
        matchings = list(Path(root_dir).glob(f'{dataset}_{domain}/*{graph_alg_name}{filter}.npy'))
        num_graph_per_domain.append(len(matchings))
        for matching in matchings:
            graph_path = str(matching)
            graph_raw = np.load(graph_path, allow_pickle=True).item()
            alg_name = graph_path.split('/')[-1].replace('.npy', '').replace("inferred_graph_", "")
            graph_sop = get_graph_sop(
                graph_raw,
                subtask_list=all_statuses,
                option_list=all_acts,
                empty_value=False if is_should else True
            )
            graph_algo_dict[alg_name] = graph_sop
            load_count += 1
        graphs[domain] = graph_algo_dict
    assert all([num_graph == num_graph_per_domain[0] for num_graph in num_graph_per_domain]), "Error. Num graph is different for each domain"
    return graphs, load_count

domains_list = list(domains)
#
print(INF_GRAPH_DIR)
is_should=False
graphs, load_count = load_graphs(domains_list, INF_GRAPH_DIR, DATASET, GRAPH_ALG_NAME, label_tuple_by_domain, is_should, filter="*")
print(f"Loaded {load_count} inferred CAN+SHDNT graphs from {len(domains)} domains")
#
print(SHD_DIR)
is_should=True
shd_sops, load_count = load_graphs(domains_list, SHD_DIR, DATASET, SHD_ALG_NAME, label_tuple_by_domain, is_should, filter="*")
print(f"Loaded {load_count} inferred SHD graphs from {len(domains)} domains")
#shd_sops = None

from multiprocess import Pool # use multiprocessing to speed up evaluation!

from util.eval_utils import dact_traj_metrics_report, dact_traj_multi_sample_metrics_report, standardize_dact
from copy import deepcopy

def eval_job(args):
    pred_params, graph_params, traj = args
    domain, model, prompt_params, temp, sampling, seed = pred_params
    prompt_style, num_shot, use_mask_prompt = prompt_params
    is_multisampling = float(temp) > 0 and ('repeat' not in sampling)
    graph_names, graph_tuples = graph_params
    graphs, neg_pcond_mats, should_sops = [], [], []
    for graph_tuple in graph_tuples:
        graph, should_sop = graph_tuple
        graphs.append(graph)
        neg_pcond_mats.append(None)
        should_sops.append(should_sop)
    
    if not isinstance(traj, tuple):
        gt_processed_label_tuple = tuple(traj)
    else:
        gt_processed_label_tuple = traj
    #print(f'In {pred_params} with multisampling={is_multisampling}')
    if is_multisampling:
        report_list = dact_traj_multi_sample_metrics_report(*gt_processed_label_tuple, graph_sop=graphs, neg_precond_mat=neg_pcond_mats, should_sops=should_sops, verbose=False)
    else:
        report_list = dact_traj_metrics_report(*gt_processed_label_tuple, graph_sop=graphs, neg_precond_mat=neg_pcond_mats, should_sops=should_sops, verbose=False)
    
    metrics_list = []
    #print(graph_names)
    for reprt, graph_name in zip(report_list,graph_names):
        if not isinstance(reprt, tuple):
            reprt = [reprt]
        for report in reprt:
            stats = report['Predicted']
            post = report['post']
            metrics = {
                'domain': domain,
                'model': model,
                'prompt': prompt_style,
                'shot': num_shot,
                'use_mask_prompt': use_mask_prompt,
                'temp': temp,
                'sampling': sampling,
                'seed': seed,
                'graph': graph_name,    
                'precision': stats['precision'],
                'recall': stats['recall'],
                'f1': stats['f1-score'],
                'support': stats['support'],
                'postprocess': post
            }
            metrics_list.append(metrics)
    return metrics_list
    
jobs = []
assert shd_sops is not None, "Error: SHOULD is empty"
for pred_params, mapped_pred_tuple in mapped_predictions.items():
    domain, model, prompt_params, temp, sampling, seed = pred_params
    #print(domain)
    #print(graphs)
    if temp > 0: # in case multi sampling, we cannot run without graph
        graph_list = list(graphs.get(domain, {}).items())
        shd_list = list(shd_sops.get(domain, {}).items())
    else:
        graph_list = list(graphs.get(domain, {}).items())
        shd_list = [('(None)', None)] + list(shd_sops.get(domain, {}).items())
    #print(graph_list)
    if len(graph_list) == 0 :
        continue
    assert len(graph_list) == 1, "Error: current code cannot handle more than one precondition"
    can_graph = graph_list[0]
    
    graph_names = [shd[0] for shd in shd_list]
    graph_tuples = [(can_graph[1], shd[1]) for shd in shd_list]
    
    graph_params = (graph_names, graph_tuples)
    mapped_pred_tuple = mapped_pred_tuple[:-2] # remove last two: all_acts, all_statuses
    jobs.append((pred_params, graph_params, mapped_pred_tuple))
print(f"# jobs={len(jobs)}")

with Pool(min(60, len(jobs))) as p:
    raw_metrics = [result for result in tqdm(p.imap(eval_job, jobs)) if result is not None]
#raw_metrics = [eval_job(job) for job in jobs]
metrics = []
for elem in raw_metrics:
    for metric_dict in elem:
        metrics.append(metric_dict)
print(f"output={len(metrics)}")


org_metrics_df = pd.DataFrame(metrics)
metrics_df = org_metrics_df.copy()

nunique = metrics_df.nunique()
cols_to_drop = nunique[nunique == 1].index
metrics_df = metrics_df.drop(cols_to_drop, axis=1)

gpt_base_performance = 0.787513 if DATASET == 'SGD' else 0.446
t5_base_performance = 0.499171 if DATASET == 'SGD' else 0.304
base_performance = (gpt_base_performance + t5_base_performance) / 2
print(f"[no-should] Mean={base_performance:.3f}, GPT={gpt_base_performance:.3f}, T5={t5_base_performance:.3f}")

rows = ['model']
columns = ['graph']
best_df = metrics_df[metrics_df['postprocess'].isin(['max'])]
display_df = best_df.pivot_table(index=rows, columns=columns, values='f1', aggfunc='mean')
print(display_df)