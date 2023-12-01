import openai
openai.api_key = open('../../apikey.token2').readlines()[0][:-1]
def getcompletion(prompt,engine = 'text-davinci-003',max_token=500):
    completion = openai.Completion.create(engine=engine,prompt=prompt,max_tokens=max_token,temperature=0)
    return completion.choices[0].text
