import tiktoken
import re

MAX_NUM_TOKEN_BY_MODEL = {
    "gpt-turbo": 4080,
    "gpt-3": 4080,
    "alpaca": 2048,
    "t5": 4080,
}

NUM_SAMPLES = 10
def num_tokens_from_prompt(prompt, model="gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    message = {'role':'user','content':prompt}
    
    if model == "gpt-3.5-turbo":  # note: future models may deviate from this
        num_tokens = 0
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
        return num_tokens
    else:
        raise NotImplementedError(f"num_tokens_from_messages() is not presently implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tok")
