# Optional Data Preprocessing

All steps here are already done for you and the resulting pre-processed files are already present in the repository under `datasets/` folder. You do not need to run these unless you want to reproduce those files)

## SGD:

1. Clone [SGD dataset repository](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) somewhere on your file system

2. Make sure you are in the `src` working directory (i.e. `cd src` from our repo’s main folder) when running all scripts below

3. Run `python3 preprocessing/preprocess_data_SGD <path_to_cloned_SGD_dataset_folder>` This should produce the files under `datasets/SGD/trajectories`

   Warning: since the train/val splits are random, the files you produce this way will have a different train/val split from the official ones. Please use the official splits from the repository for all experiments to ensure exact reproduction of results. Unfortunately we lost the seed value used to generate the official train/val splits, so we could not exactly reproduce the splits.

4. Run `python3 preprocessing/create_GT_label_for_action_prediction.py SGD` to create the files under `datasets/SGD/action_prediction_gt_labels_train+val` and `datasets/SGD/action_prediction_gt_labels_test_only`

## MultiWOZ:

1. Download [MultiWOZ dataset zip folder](https://github.com/lexmen318/MultiWOZ-coref/blob/main/MultiWOZ2_3.zip) , unzip the folder somewhere on your file system

2. Make sure you are in the `src` working directory (i.e. `cd src` from our repo’s main folder) when running all scripts below

3. Run `python3 preprocessing/preprocess_data_multiwoz_multi.py <path_to_unzipped_multiwoz_folder>` and then run `python3 preprocess_data_multiwoz_single.py <path_to_unzipped_multiwoz_folder>` to produce the files under `datasets/MultiWOZ/trajectories`. Note that you must run `multi` before `single`.

4. Run `python3 preprocessing/create_GT_label_for_action_prediction.py MultiWOZ` to create the files under `datasets/MultiWOZ/action_prediction_gt_labels_train+val` and `datasets/MultiWOZ/action_prediction_gt_labels_test_only`
