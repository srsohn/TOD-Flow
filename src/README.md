These are the list of scripts and their functional description.
The exact command will be added soon!

|   | input                                     | output                  | filename                | author |
|---|-------------------------------------------|-------------------------|-------------------------|--------|
| 1 | raw_data, attr_mapping                    | ilp_data                | run_data_relabling.py   | YL     |
| 2 | ilp_data                                  | inferred_graph          | run_graph_prediction.py | SR, AL |
| 3 | inferred_graph, GT_mapped_graph, ilp_data | accuracy, coverage      | run_graph_eval.py       | SR     |
| 4 | ilp_data, raw_data                        | Visualize               | visualize_data.ipynb    | YL     |
| 5 | graph (GT/inferred)                       | simulated_dialogue_data | run_dialogue_manager.py | AL, SR |

## Inferring a graph (with run_graph_prediction.py)

Example command:
```bash
python run_graph_prediction.py --visualize --data_path ../datasets/SGD/trajectories/Restaurants_1_trajectories.json --task Restaurants_1_Search_Reserve --gt_graph_path ../datasets/SGD/gt_graph/Restaurants_1_gt_graph.npy
```

Usage information: `python run_graph_prediction.py`

By default, `run_graph_prediction.py` will write inferred graph to the `../ilp_out/<dataset>_<task>` directory.

If the optional visualize flag is given, the visualizations will be written to `../ilp_out/visualize/<dataset>_<task>`.

If the optional `gt_graph_path` argument is given, the predicted graph will be evaluated against the provided ground truth graph.
