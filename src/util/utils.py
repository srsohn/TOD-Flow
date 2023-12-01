import os
from os import listdir
from os.path import isfile, join
import json
import random
from collections import defaultdict
import numpy as np
import scipy.io
import torch
import math
import pickle as pkl
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from collections import Counter
from string import punctuation
from torch.nn import CrossEntropyLoss
from scipy.linalg import block_diag

def preproc(summary):
    summary = summary.split('\n')
    summary = [s for s in summary if s]
    steps = []
    for step in summary:
        step = step.strip(punctuation)
        for i, c in enumerate(step):
            if c.isalpha():
                steps.append(step[i:])
                break
    return steps

def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return idx, np.column_stack(np.unravel_index(idx, a.shape))

def compute_similarity(asr, summary, model, cross_encoder=True):

    if cross_encoder:
        scores = np.array([model.predict([(x, y) for y in summary]) for x in asr])

    else:
        sent_embs = model.encode(asr, show_progress_bar=False)
        subtask_embs = model.encode(summary, show_progress_bar=False)
        # scores = util.cos_sim(sent_embs, subtask_embs)
        scores = util.dot_score(sent_embs, subtask_embs)

    best_match = np.argmax(scores, axis=1).tolist()

    return scores, best_match

def community_detection_with_scores(scores, threshold=0.75, min_community_size=10, init_max_size=1000):
    """
    Function for Fast Community Detection
    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.
    """

    init_max_size = min(init_max_size, len(scores))

    # Minimum size for a community
    top_k_values, _ = scores.topk(k=min_community_size, largest=True)

    # Filter for rows >= min_threshold
    extracted_communities = []
    for i in range(len(top_k_values)):
        if top_k_values[i][-1] >= threshold:
            new_cluster = []

            # Only check top k most similar entries
            top_val_large, top_idx_large = scores[i].topk(k=init_max_size, largest=True)
            top_idx_large = top_idx_large.tolist()
            top_val_large = top_val_large.tolist()

            if top_val_large[-1] < threshold:
                for idx, val in zip(top_idx_large, top_val_large):
                    if val < threshold:
                        break

                    new_cluster.append(idx)
            else:
                # Iterate over all entries (slow)
                for idx, val in enumerate(scores[i].tolist()):
                    if val >= threshold:
                        new_cluster.append(idx)

            extracted_communities.append(new_cluster)

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for community in extracted_communities:
        add_cluster = True
        for idx in community:
            if idx in extracted_ids:
                add_cluster = False
                break

        if add_cluster:
            unique_communities.append(community)
            for idx in community:
                extracted_ids.add(idx)

    return unique_communities

def get_embeddings(steps, model, tokenizer):

    steps = [' ' + step + '.' for step in steps]
    lens = [len(tokenizer.encode(step)) - 2 for step in steps]
    total_len = len(tokenizer.encode(''.join(steps))) - 2

    assert np.sum(lens) == total_len

    with torch.no_grad():
        inputs = tokenizer(''.join(steps), return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state.squeeze()

    embs = last_hidden_states[1:-1].split(lens, dim=0)
    embs = [emb.mean(0).cpu().numpy() for emb in embs]
    return embs

def write_json(data, fname):
    with open(fname, 'w') as fp:
        json.dump(data, fp, indent=2)

def load_json(fname):
    with open(fname, 'r') as fp:
        data = json.load(fp)
    return data

def loss_per_sample(inputs, logits, attention_mask, length=None, normalize=False):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    # loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    loss = loss.view(shift_logits.size(0), shift_logits.size(1))
    loss = loss * attention_mask[..., 1:]
    if length is not None:
        loss = loss.squeeze()[-length:].sum()
        if normalize:
            loss = loss / length
        return loss
    return loss.sum(axis=1)

def compute_logprobs(hyps, model, tokenizer, sequential=False, lens=None, normalize=False, device='cuda:0'):

    losses = []
    for i, hyp in enumerate(hyps):
        inputs = tokenizer(hyp, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            if lens is not None:
                per_sample_loss = loss_per_sample(inputs['input_ids'], outputs.logits, inputs['attention_mask'], length=lens[i], normalize=normalize)
            else:
                per_sample_loss = loss_per_sample(inputs['input_ids'], outputs.logits, inputs['attention_mask'])
            losses.append(per_sample_loss.item())
            # import pdb; pdb.set_trace()
    losses = np.array(losses).reshape(-1, 1)
    return -losses

def beam_search(
    prompt,
    choices,
    model,
    tokenizer,
    choice_mask,
    max_subgoals=10,
    beam_size=10,
    sequential=True,
    length_normalize=False,
    prior=None,
    length_penalty=False,
    length_penalty_thresh=0,
    reference_length=0,
    device='cuda',
    ):

    partial_hyps_prior = []
    # if prior is not None:
    #     partial_hyps_prior = [prior]
    partial_hyps = [prompt]
    partial_steps = []
    partial_lps = np.array([[0]])

    # choices_lens = [len(tokenizer.encode(choice)) for choice in choices]
    base_len = len(tokenizer.encode(prompt))

    count = 0
    while count == 0 or (partial_steps[:, -1] != len(choices)).any():
        count += 1
        full_hyps = []
        full_hyps_prior = []

        next_choices = [f'\n{count}. {step}' for step in choices]
        next_choices.append(f'\n{count}. Done')
        choices_lens = [len(tokenizer.encode(choice)) for choice in next_choices]

        for partial_hyp in partial_hyps:
            full_hyps.extend([f'{partial_hyp}{choice}' for choice in next_choices])
        # for partial_hyp in partial_hyps_prior:
        #     full_hyps_prior.extend([f'{partial_hyp}{choice}' for choice in next_choices])

        if length_normalize:
            logprobs = compute_logprobs(full_hyps, model, tokenizer, sequential=sequential, lens=len(partial_hyps) * choices_lens, normalize=True, device=device)
            # lens = [len(tokenizer.encode(hyp)) - base_len for hyp in full_hyps]
            # logprobs = compute_logprobs(full_hyps, model, tokenizer, sequential=sequential, lens=lens, normalize=True, device=device)
        else:
            logprobs = compute_logprobs(full_hyps, model, tokenizer, sequential=sequential, device=device)
            # if prior is not None:
            #     logprobs_prior = compute_logprobs(full_hyps_prior, sequential=sequential)
            #     logprobs = 2*logprobs - logprobs_prior

        logprobs = logprobs.reshape(len(partial_hyps), -1)
        logprobs = partial_lps + logprobs if length_normalize else logprobs
        # logprobs[:, -1] = logprobs[:, -1] - abs(count - reference_length) * length_penalty

        eos = np.zeros((beam_size, 1)).astype(int)

        if length_penalty and count < reference_length - length_penalty_thresh:
            logprobs[:, -1] = -1e6

        # Disallow empty sequences
        if count == 1:
        # if count < min_subgoals:
            logprobs[:, -1] = -1e6
        else:
            # Disallow repeated subtasks
             mask = choice_mask[partial_steps.astype(int)]
             mask = mask.sum(1) > 0
             logprobs = -1e6 * mask + logprobs * (1 - mask)

        ##### Post processing #####
        if count > 1:
            # 1) If partial hyp is already complete, just copy over the previous logprobs
            eos = partial_steps[:, [-1]] == len(choices) # Find which hyps are already complete
            logprobs = eos * partial_lps + (1 - eos) * logprobs

            logprobs = np.concatenate([
                logprobs[:, :-1] * (1 - eos) + -1e6 * eos,
                logprobs[:, [-1]]
            ], axis=1)

        argsort_inds, argsort_inds_unravel = k_largest_index_argsort(logprobs, beam_size)
        hyp_idx, step_idx = argsort_inds_unravel[:, 0], argsort_inds_unravel[:, 1]

        # if length_normalize:
        #     partial_lps = partial_lps[hyp_idx] + logprobs[hyp_idx, step_idx].reshape(-1, 1)
        # else:
        #     partial_lps = logprobs[hyp_idx, step_idx].reshape(-1, 1)
        partial_lps = logprobs[hyp_idx, step_idx].reshape(-1, 1)

        # Top scoring hyps
        partial_hyps = [full_hyps[i] for i in argsort_inds]
        # if prior is not None:
        #     partial_hyps_prior = [full_hyps_prior[i] for i in argsort_inds]

        # if partial hyp is already complete, set the correct step id
        step_idx = (1 - eos[hyp_idx]) * step_idx.reshape(-1, 1) + eos[hyp_idx] * len(choices)

        # Keep track of step ids
        if count == 1:
            partial_steps = step_idx
        else:
            partial_steps = partial_steps[hyp_idx, :]
            partial_steps = np.concatenate((partial_steps, step_idx), axis=1)

        # print(partial_steps)

        if count == max_subgoals:
            break

    idx = np.argmax(partial_lps)
    steps = partial_steps[idx]

    labels = [step for step in steps if step != len(choices)]
    steps = [choices[label] for label in labels]

    return steps, partial_hyps, partial_lps, partial_steps

def load_model(model_name, opt_weights_path=None, fp16=True, device_map=None):
    from transformers import AutoTokenizer

    if 'opt-175b' in model_name:
        model, device_map = get_opt_model(opt_weights_path, fp16=fp16)
        tokenizer = AutoTokenizer.from_pretrained('facebook/opt-30b')
        device = device_map['lm_head']
    elif 'opt-30b' in model_name:
        # from huggingface_hub import snapshot_download
        # weights_path = snapshot_download(model_name)
        # model, device_map = get_opt_model(weights_path, fp16=fp16, device_map=device_map)
        model, device_map = get_opt_model(opt_weights_path, fp16=fp16, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif 'gpt-j' in model_name:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    else:
        from transformers import AutoModelForCausalLM
        if 'opt' in model_name:
            model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def load_key_steps(task_name, key_steps):
    if key_steps == 'gt':
        task_label_text, _ = load_config(task_name)
        clusters = [[choice] for choice in task_label_text]
    elif key_steps == 'infer':
        clusters = load_json(f'Dataset/key_steps/{task_name}.json')
    else:
        task_label_text = load_json(key_steps)
        clusters = [choice if isinstance(choice, list) else [choice] for choice in task_label_text]
    return clusters

def load_clusters(clusters, max_cand_per_cluster=100, termination_symbol=True):
    clusters = [cluster[:max_cand_per_cluster] for cluster in clusters]
    choices = [x for cluster in clusters for x in cluster]
    choice_to_cluster_id = dict()

    for i, cluster in enumerate(clusters):
        for sent in cluster:
            if sent in choice_to_cluster_id:
                import pdb; pdb.set_trace()
                assert False
            choice_to_cluster_id[sent] = i

    cluster_matrix = [np.ones((len(cluster), len(cluster))) for cluster in clusters]
    if termination_symbol:
        cluster_matrix = cluster_matrix + [np.array([0])]
    choice_mask = block_diag(*cluster_matrix)

    return choices, choice_to_cluster_id, choice_mask

def label_subtasks(subtasks, labels, label_to_id, label_mask, encoder, sim_thresh=0, cross_encoder=False):

    if cross_encoder:
        scores1 = np.array([encoder.predict([(x, y) for y in labels]) for x in subtasks])
        scores2 = np.array([encoder.predict([(y, x) for y in labels]) for x in subtasks])
        scores = 0.5 * (scores1 + scores2)
    else:
        subtask_embeddings = encoder.encode(subtasks, show_progress_bar=False)
        label_embeddings = encoder.encode(labels, show_progress_bar=False)
        scores = util.cos_sim(subtask_embeddings, label_embeddings)
        scores = scores.numpy()

    chosen_subtasks = set()
    chosen_labels = set()
    assignments = set()

    max_score = scores.max()
    while max_score > sim_thresh:
        x, y = np.where(scores == max_score)
        a, b = x[0], y[0]
        chosen_subtasks.add(a)
        chosen_labels.add(b)
        assignments.add((a, b))
        scores[a, :] = 0
        # scores[:, b] = 0
        scores = (1 - label_mask[[b]]) * scores
        max_score = scores.max()

    assignments = sorted(list(assignments), key=lambda x: x[0])
    preds = [x[1] for x in assignments]
    # preds = [int(x) for x in preds]

    predicted_subtasks = [labels[label_idx] for label_idx in preds]
    predicted_labels = [label_to_id[labels[label_idx]] for label_idx in preds]

    # preds = scores.argmax(-1)
    # return preds.tolist()

    return predicted_labels, predicted_subtasks

def find_cliques(scores, threshold, min_clique_size=5):
    import networkx as nx
    graph = nx.from_numpy_matrix(scores > threshold, parallel_edges=False, create_using=None)
    cliques = nx.find_cliques(graph)
    cliques = sorted([c for c in cliques], key=lambda x: len(x), reverse=True)
    cliques = [c for c in cliques if len(c) >= min_clique_size]
    return cliques

def get_final_cliques(
    predictions,
    sentence_to_sequences,
    encoder,
    thresholds,
    inter_cluster_threshold,
    min_cluster_size=5
    ):
    from sentence_transformers import util

    sentences = [s for pred in predictions for s in pred]
    # embeddings = [emb for steps in predictions for emb in get_embeddings(steps)]
    embeddings = encoder.encode(sentences, show_progress_bar=True)
    assert len(sentences) == len(embeddings)
    scores = util.cos_sim(embeddings, embeddings)
    scores = scores.numpy()

    all_cliques = []
    for threshold in thresholds:
        cliques = find_cliques(scores, threshold, min_cluster_size)
        all_cliques.append(cliques)

    final_cliques = []
    extracted_ids = set()
    extracted_sents = set()
    final_cliques_sentences = []

    for cliques in all_cliques:
        for clique in cliques:
            clique_sents = [sentences[c] for c in clique]

            # if extracted_sents & set(clique_sents):
            #     continue

            if extracted_ids & set(clique):
                continue

            clique_sentences = set().union(*[sentence_to_sequences[sentences[c]] for c in clique])
            no_overlap = False
            for sents, final_clique in zip(final_cliques_sentences, final_cliques):
                if not sents & clique_sentences: # No overlapping sequences
                    # Check inter cluster similarity
                    if scores[final_clique[0]][clique[0]] > inter_cluster_threshold:
                        no_overlap = True
                        break
            if no_overlap:
                continue

            # high_cluster_overlap = False
            # for final_clique in final_cliques:
            #     if scores[final_clique[0]][clique[0]] > intra_cluster_threshold:
            #         high_cluster_overlap = True
            #         break
            # if high_cluster_overlap:
            #     continue

            final_cliques.append(clique)
            clique_sentences = set()
            for c in clique:
                clique_sentences = clique_sentences | sentence_to_sequences[sentences[c]]
            final_cliques_sentences.append(clique_sentences)

            extracted_ids = extracted_ids | set(clique)
            extracted_sents = extracted_sents | set(clique_sents)

    return final_cliques, final_cliques_sentences

def interval_overlap(x, y):
    if x[1] <= y[0]:
        return 0
    if x[0] >= y[1]:
        return 0

    return min(x[1], y[1]) - max(x[0], y[0])

def interval_distance(x, y):
    return abs((x[0] + x[1])/2 - (y[0] + y[1])/2)

def find_nearest_interval(x, y):
    min_distance = 1e6
    min_distance_idx = None

    overlap_interval_idxs = []

    for count, i in enumerate(y):

        dist = interval_distance(x, i)
        if dist < min_distance:
            min_distance_idx = count
            min_distance = dist

        overlap = interval_overlap(x, i)
        if overlap:
            overlap_interval_idxs.append(count)

    if overlap_interval_idxs:
        return overlap_interval_idxs

    return [min_distance_idx]

def find_cluster(descriptions, reference, encoder, topk=5, thresh=0.8, reference_nn=True):

    if reference_nn:
        embeddings = encoder.encode(descriptions)
        embedding_reference = encoder.encode([reference])
        scores = util.cos_sim(embedding_reference, embeddings)
        scores = scores.squeeze(0)
        idxs = np.argsort(scores).tolist()[::-1]
        cluster = [reference]
        for i, idx in enumerate(idxs):
            if i >= topk:
                break
            if scores[idx].item() < thresh:
                break
            cluster.append(descriptions[idx])
    else:
        embeddings = encoder.encode(descriptions)
        scores = util.cos_sim(embeddings, embeddings)
        # scores = scores * (1 - np.eye(scores.shape[0]))
        # topk = min(topk, scores.shape[1])
        # avg_sim = scores.topk(topk, axis=1)[0].mean(axis=1)
        # idxs = np.argsort(avg_sim).tolist()[::-1][:topk]
        # cluster = [descriptions[idx] for idx in idxs if avg_sim[idx] > thresh]

        # embeddings = encoder.encode(descriptions)
        # mean_embedding = embeddings.mean(axis=0, keepdims=True)
        # scores = util.cos_sim(mean_embedding, embeddings)
        # scores = scores.squeeze(0)
        # import pdb; pdb.set_trace()

        cluster_center = np.argmax(scores.mean(axis=1))
        scores = scores[cluster_center]

        idxs = np.argsort(scores).tolist()[::-1][:topk]
        # cluster = [descriptions[idx] for idx in idxs if scores[idx] > thresh]
        cluster = [descriptions[idx] for idx in idxs]

    return cluster

def eliminate_duplicates(clusters):

    instance_count = Counter([x for cluster in clusters for x in set(cluster)])
    remove_instances = set([instance for instance in instance_count if instance_count[instance] > 1])
    clusters = [[x for x in cluster if x not in remove_instances] for cluster in clusters]

    return clusters

def find_cluster_v2(label_descriptions, gtlabels, encoder, topk=5, inter_cluster_thresh=0.7):

    clusters = []
    for i in range(len(label_descriptions)):
        clusters.append(label_descriptions[i])

    clusters = eliminate_duplicates(clusters)
    for i, cluster in enumerate(clusters):
        if not cluster:
            cluster.append(gtlabels[i])

    sentences = [sent for cluster in clusters for sent in cluster]
    embeddings = encoder.encode(sentences)
    scores = util.cos_sim(embeddings, embeddings)
    scores = scores * (1 - np.eye(scores.shape[0]))
    diag_mats = [np.ones((len(cluster), len(cluster))) for cluster in clusters]
    block_mask = block_diag(*diag_mats)

    intra_cluster_sim = scores * block_mask
    inter_cluster_sim = scores * (1 - block_mask)

    # topk_sim = 1
    # topk_cluster = 1
    # inter_cluster_thresh = 0.7

    # avg_intra_sim = intra_cluster_sim.topk(topk_sim, axis=1)[0].mean(axis=1)
    # avg_inter_sim = inter_cluster_sim.topk(topk_sim, axis=1)[0].mean(axis=1)

    avg_intra_sim = intra_cluster_sim.mean(axis=1)
    # avg_inter_sim = inter_cluster_sim.mean(axis=1)
    # avg_inter_sim = inter_cluster_sim.max(axis=1)[0]

    avg_intra_sim_split = avg_intra_sim.split([len(cluster) for cluster in clusters])
    cluster_centers = [np.argmax(split) for split in avg_intra_sim_split]
    cluster_centers_oh = []
    for center, cluster in zip(cluster_centers, clusters):
        center_oh = np.zeros(len(cluster))
        center_oh[center] = 1
        cluster_centers_oh.append(center_oh)
    cluster_centers_oh = np.concatenate(cluster_centers_oh)
    avg_inter_sim = inter_cluster_sim * np.expand_dims(cluster_centers_oh, 0)
    avg_inter_sim_max = avg_inter_sim.max(axis=1)[0]

    # sentence_quality = avg_intra_sim - avg_inter_sim
    # sentence_quality = avg_intra_sim / torch.clamp(avg_inter_sim, min=0.2)
    # sentence_quality = avg_intra_sim / avg_inter_sim
    # sentence_quality = avg_intra_sim / torch.clamp(avg_inter_sim, min=0.7)
    # sentence_quality = avg_intra_sim * (avg_inter_sim < inter_cluster_thresh)
    sentence_quality = avg_intra_sim * (avg_inter_sim_max < inter_cluster_thresh)
    cluster_sentence_quality = sentence_quality.split([len(cluster) for cluster in clusters])
    cluster_best_sentences = [np.argsort(sentence_quality).tolist()[::-1][:topk] for sentence_quality in cluster_sentence_quality]
    new_clusters = [[cluster[idx] for idx in idxs] for idxs, cluster in zip(cluster_best_sentences, clusters)]

    return new_clusters

def read_data(datapath):
    summaries = []
    filenames = [x for x in os.listdir(datapath) if 'txt' in x]
    for filename in filenames:
        with open(f'{datapath}/{filename}', 'r') as f:
            data = f.read()
        # data = preproc(data)
        summaries.append(data)
    return summaries
