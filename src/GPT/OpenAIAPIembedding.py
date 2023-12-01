import openai
openai.api_key = open('../../apikey.token2').readlines()[0][:-1]
def getembedding(querys):
    embeds = openai.Embedding.create(model='text-embedding-ada-002',input=querys)
    return [e['embedding'] for e in embeds['data']]