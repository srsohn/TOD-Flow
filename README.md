# TOD-flow
This repo is the implementation of EMNLP 2023 conference paper "TOD-Flow: Modeling the Structure of Task-Oriented Dialogues".

> [!NOTE]  
> You can skip the step with (Optional) since the output is pre-generated in the repo. If you want a full reproduction of the result, you can follow the instruction to regenerate those outputs.

## Setup

You must complete all steps here before attempting to run any experiments!


### Repository and environment setup


To setup the repository, run the following commands:

```
git clone https://github.com/srsohn/dialog_flow.git
cd dialog_flow
conda create --name todflow python=3.10 
conda activate todflow
pip install -e .
```

Some optional parts of the experiment reproduction involves using OpenAI API to access ChatGPT. To use the API, you must have an OpenAI API key, and set the environment variable `OPENAI_API_KEY` to be your API key.

### Downloading data

We could not include the content under "datasets" and "outputs" folder due to its size. Before running any of our scripts, please download [datasets.tar](https://drive.google.com/file/d/1RQbHcjt3LHyrbvnYiIz37b_dwDg7Nyb_/view?usp=sharing) and move it into `datasets` folder, and untar it there. Also, please download [outputs.tar](https://drive.google.com/file/d/1NC7lmnG2Oc4uZQmI3x7flXKPUX8RBO4i/view?usp=sharing) and move it into the `outputs` folder, and untar it there.


## Dialogue policy experiment on SGD and MultiWOZ

1. (Optional) Preprocess data: The preprocessed data is pre-generated in `datasets/`. For full reproduction of result: see instruction in `src/preprocessing/README.md`
2. (Optional) Run base LLM (GPT-3.5 and FLAN-T5): The LLM prediction is pre-generated in `outputs/`. Make sure you are in working directory `dialog_flow/src`. If you want to reproduce our result, you can run the following script:
    ```
    bash ../my_script/MW_run_GPT_T5.sh flan-t5-xxl
    OPENAI_API_KEY=<openai-api-key> bash ../my_script/MW_run_GPT_T5.sh gpt-turbo
    bash ../my_script/SGD_run_GPT_T5.sh flan-t5-xxl
    OPENAI_API_KEY=<openai-api-key> bash ../my_script/SGD_run_GPT_T5.sh gpt-turbo
    ```

    > This process may take hours and cost around **$20** for GPT-turbo-3.5.
    > 
    > If you run flan-t5, make sure that the pip-installed torch version is compatible with your cuda driver version. If not, you may need to manually reinstall a torch package that is compatible with your cuda driver version following instructions from [official PyTorch Website](https://pytorch.org/get-started/previous-versions/).

3. (Optional) Map LLM predictions to GT actions by running the following for both SGD and MultiWOZ (Make sure you are in working directory `dialog_flow/src`.): 
    ```
    python3 map_prediction_to_GT_action_final.py MultiWOZ
    python3 map_prediction_to_GT_action_final.py SGD
    ```
    The resulting files of completing steps 2 and 3 should have similar structure as the pre-generated LLM predictions in `outputs/`.
4. (Optional) Infer TOD-flow graph. (Make sure you are in working directory `dialog_flow/src`.)
    ```
    bash ../my_script/generate_MWF_graph_csilp_final.sh
    bash ../my_script/generate_MWF_graph_shdilp_final.sh
    bash ../my_script/generate_SGD_graph_csilp_final.sh
    bash ../my_script/generate_SGD_graph_shdilp_final.sh
    ```
    The resulting files should be graphs under `graphs/` folder.
5. Evalulate graph-conditioned dialogue policy by running the following: (Make sure you are in working directory `dialog_flow/src`.)
    ```
    python3 eval_GCDM_should.py SGD
    python3 eval_GCDM_should.py MultiWOZ
    ```
    At the end of running each script, the resulting scores should match performance of "Ours" in Table 3 of our paper.

## End-to-end experiment on MultiWOZ

1. Clone the [MultiWOZ_Evaluation](https://github.com/Tomiinek/MultiWOZ_Evaluation.git) repository at the same directory level as our TOD-flow repository. (If installed elsewhere, please modify line 11 of `src/endtoend/run_grid_search.py` to your cloned location)

    Then please change line 120 in `mwzeval/metrics.py` of that repository from `if not has_domain_predictions(input_data):` to `if True:`. We do this to ensure that we always use ground truth domains for evaluation and to get consistent results.
 
2. (Optional) Preprocess data: The preprocessed data is pre-generated in `datasets/`. For full reproduction of result: see instruction in `./preprocessing/README.md`

3. (Optional) Run the base end-to-end dialogue models (GALAXY, HDNO, HDSA): The detailed process of producing predictions for each method is too complicated, as it involves cloning each method's official repository and manually making edits to parts of their codes to record multi-sampled results. We therefore omit the details for this step and recommend to simply use the pre-generated predictions in `datasets/MultiWOZ/e2e`
    ```
    Galaxy: datasets/MultiWOZ/e2e/galaxy_7full_pred.json
    HDSA: datasets/MultiWOZ/e2e/hdsa_7new_pred.json
    HDNO: datasets/MultiWOZ/e2e/hdno_7_pred.json
    ```

4. Set your working directory to `src/`. All commands below should be executed from that working directory.

5. Infer TOD-flow graph for end-to-end setting
    ```
    bash ../my_script/generate_MWF_graph_csilp_E2E.sh
    bash ../my_script/generate_MWF_graph_shdilp_E2E.sh
    ```

6. (Optional) Annotate the base model predictions using ChatGPT as NLU model (i.e. parsing predicted responses into actions).
    ```
    OPENAI_API_KEY=<openai-api-key> bash endtoend/run_all_GPT.sh <method_name>
    OPENAI_API_KEY=<openai-api-key> bash endtoend/postprocess_all_GPT.sh <method_name>
    ```
    Where method name is one of `GALAXY` `HDNO` `HDSA`. The above steps may take hours and cost dozens of dollars as payment to OpenAI, with non-deterministic results. The results will be stored in `datasets/MultiWOZ/e2e/<method_name>`, where the `xxx_actions.json` stores the raw action predictions for each utterance from GPT, and `xxx_predform.json` stores them in standardized format for graph filtering.

    > This process may take hours and cost around **$150** for GPT-turbo-3.5.
8. Evalulate graph-conditioned end-to-end dialogue model: run the command below:
    ```
    python3 endtoend/run_grid_search.py <method_name>
    ```
    Where <method_name> refers to one of `GALAXY`, `HDNO`, `HDSA`, and `GALAXYSTAR`. The program will print the best result and hyperparameters at the end.

## Reference

```
TODO when paper out
```
