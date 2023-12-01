import openai
import time
from joblib import Memory
from GPT.GPTutils import NUM_SAMPLES
import os

try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
except:
    try:
        openai.api_key = open('../../apikey.token2').readlines()[0][:-1]
    except:
        print('warning: openai api key not set')

memory = Memory("cachedir", verbose=0)


@memory.cache
def getcompletion(prompt, model, temperature,maxlen = 50):
    if 'turbo' in model:
        engine = 'gpt-3.5-turbo'
    if temperature == 0:
        completion = openai.ChatCompletion.create(model = engine, max_tokens=maxlen, messages = [{'role':'user','content':prompt}], temperature=0)
        c =  completion.choices[0].message.content
        return c
    else:
        completion = openai.ChatCompletion.create(model = engine, max_tokens=maxlen, messages = [{'role':'user','content':prompt}], n=NUM_SAMPLES, temperature=temperature)
        c = [completion.choices[i].message.content for i in range(NUM_SAMPLES)]
        return c

def batch_getcompletion(prompt_list, engine = 'gpt-3.5-turbo',max_token=50,multisample = None):
    output_list = []
    for prompt in prompt_list:
        time.sleep(1)
        output = getcompletion(prompt, engine, max_token, multisample)
        output_list.append(output)
    return output_list


