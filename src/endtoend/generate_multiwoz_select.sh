#!/bin/bash

python3 endtoend/Galaxy_graph_sample.py --gt_path ../datasets/MultiWOZ/action_prediction_gt_labels_test_only/$1_labels.json --pred_path ../datasets/MultiWOZ/e2e/$2/$1_predform.json --out_path ../datasets/MultiWOZ/e2e/$2/$1_select.json --graph_mask $7/MultiWOZ_$1/inferred_graph_CSILP_fgam=0.95_bgam=0.9_pos=$3_dep=6_leaf=0.01.npy --shd_graph_mask $8/MultiWOZ_$1/inferred_graph_SHDILP_bw=4_bd=4_cp=0.01_mins=$4.npy --can_factor $5 --shd_thresh $6 --method $9

