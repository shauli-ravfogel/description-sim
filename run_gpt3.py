import numpy as np
# import pca
from sklearn.decomposition import PCA
import pickle
import random
from datasets import load_dataset
import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import random
import spacy
import os
import openai
import tqdm
import time
from sklearn.utils import shuffle

openai.api_key = ""
CHATGPT=False

def get_generation(prompt, sentence_query, neighbor, n=1, description_query= None):
    if not CHATGPT:

        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt.format(sentence=sentence_query) if description_query is None else prompt.format(sentence=sentence_query, description=description_query),
        temperature=np.random.random()*0.7,
        max_tokens=300,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\"\"\""],
        n=n
        )

        return [(response["choices"][i]["text"]) for i in range(len(response["choices"]))] 
    else:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                            messages=[{"role": "user", 
                                    "content": prompt.format(sentence=sentence_query) if description_query is None else prompt.format(sentence=sentence_query, description=description_query),}]
        )
  
        return [(response["choices"][i]["message"]["content"]) for i in range(len(response["choices"]))] 


with open("pubmed.txt", "r") as f:
    pubmed_data = f.readlines()
pubmed_data = [s.strip() for s in pubmed_data]
with open("wiki_sents_5m_v2.txt", "r") as f:
    wiki_data = f.readlines()

with open("sents_bookscorpus_shuffled_1m.txt", "r") as f:
    book_data = f.readlines()
    book_data = [s.strip() for s in book_data]

wiki_data = [d.strip() for d in wiki_data]
good_idx = np.array([i for i in range(len(wiki_data)) if len(wiki_data[i].split(" ")) > 12 and len(wiki_data[i].split(" ")) < 40])
wiki_data = [wiki_data[i] for i in good_idx]
device =  "cuda:0" if torch.cuda.is_available() else "cpu"
wiki_prompt="Let's write abstract descriptions of sentences. Example:\nSentence: Pilate 's role in the events leading to the crucifixion lent themselves to melodrama , even tragedy , and Pilate often has a role in medieval mystery plays .\nDescription: A description of a historical religious figure's involvement in a significant event and its later portrayal in art.\nNote: Descriptions can differ in the level of abstraction, granularity and the part of the sentence they focus on. Some descriptions neeed to be abstract, while others should be concrete and detailed.\nFor the following sentence, write up 5 good and stand-alone, independent descriptions and 5 bad descriptions (which may be related, but are clearly wrong). Output a json file with keys 'good', 'bad'.\nSentence: {sentence}\nStart your answer with a curly bracket."
pubmed_prompt ="Let's write abstract descriptions of sentences. Example:\nSentence: Regulatory T (Treg) cells have an immunosuppressive function and highly express the immune checkpoint receptor PD-1 in the tumor microenvironment; however, the function of PD-1 in tumor-infiltrating (TI) Treg cells remains controversial .\nDescription: description of the uncertainty regarding the role of a specific subtype of regulatory immune cells on tumor growth.\nNote: Descriptions can differ in the level of abstraction, granularity and the part of the sentence they focus on. Some descriptions neeed to be abstract, while others should be concrete and detailed.\nFor the following sentence, write up 5 good and stand-alone, independent descriptions and 5 bad descriptions (which may be related, but are clearly wrong). Output a json file with keys 'good', 'bad'.\nSentence: {sentence}\nStart your answer with a curly bracket."
bookscorpus_prompt="Let's write abstract descriptions of sentences. Example:\nSentence: `` as did i , '' sonia heard said from the direction of the door , her eyes shifted from the beautiful wedding gown and fabulous shoes to the door and she saw leah standing there . .\nDescription: A woman's gaze shifts from admiring fashionable items to a newly arrived person.\nNote: Descriptions can differ in the level of abstraction, granularity and the part of the sentence they focus on. Some descriptions neeed to be abstract, while others should be concrete and detailed.\nFor the following sentence, write up 5 good and stand-alone, independent descriptions and 5 bad descriptions (which may be related, but are clearly wrong). Output a json file with keys 'good', 'bad'.\nSentence: {sentence}\nStart your answer with a curly bracket."

results = []
n=1
pubmed=False
wiki=False
five_million = True
if pubmed:
    prompt = pubmed_prompt
    data = pubmed_data
    X = None
elif wiki:
    prompt = wiki_prompt
    data = wiki_data
    X = None
else:
    prompt = bookscorpus_prompt
    data = book_data
    X = None

random.seed(0)

data = shuffle(data, random_state=0)

for i in tqdm.tqdm(range(50000)):
    query = data[i]
    try:
        generations = get_generation(prompt, query, None, n)
        json_dict = eval(generations[0])
    except Exception as e:
        print("exception", e)
        time.sleep(5)
        continue
    json_dict["sentence"] = query
    results.append(json_dict)

    if i % 10 == 0:
        print(results[-1])
        print("=====================================")
    if i % 50 == 0 and i > 0:
        with open("sims_part1_{}_no_pairs.pickle".format("pubmed" if pubmed else "wiki" if wiki else "bookcorpus"), "wb") as f:
            pickle.dump(results, f)


