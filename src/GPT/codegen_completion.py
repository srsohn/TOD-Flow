import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from joblib import Memory
memory = Memory("cachedir", verbose=0)
#HF_CACHE = '/mnt/disks/sdb/llajan/.cache/huggingface/hub'

_model = None# cache load the LM when needed
_tokenizer = None

@memory.cache
def getcompletion(prompt, model="codegen-2B-multi", max_length=2048, output_len=50):
    global _model, _tokenizer
    device = torch.device("cuda:0")
    if _model is None:
        print('loading LLM..')
        _tokenizer = AutoTokenizer.from_pretrained(f"Salesforce/{model}")
        _model = AutoModelForCausalLM.from_pretrained(f"Salesforce/{model}").to(device)
        print('Done.')  
    
    input_ids = _tokenizer(prompt, return_tensors="pt").input_ids
    input_ids_len = input_ids.shape[1]
    assert input_ids_len < max_length - output_len, f"input length={input_ids_len} >= {max_length} - {output_len}"
    generate_len = input_ids_len + output_len
    input_ids = input_ids.to(device)
    sample = _model.generate(input_ids, temperature=0, max_length=generate_len, use_cache=True)
    #last_hidden_states = sample.last_hidden_state
    return _tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])