import streamlit as st
import transformers
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

METHOD = "mean"

def encode(model, tokenizer, sentence, device):
    # get the mpnet representation for the sentence
    tokens = tokenizer.tokenize(sentence)
    input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    features = model(input_ids)[0]
    features = features.mean(axis=1)
    return features

def encode_cls(model, tokenizer, sentence, device, pooling = "cls"):
    # get the mpnet representation for the sentence
    tokens = tokenizer.tokenize(sentence)
    input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    features = model(input_ids)[0]
    if pooling == "cls":
        features = features[:,0,:]
    else:
        features = features.mean(axis=1)
    return features

def encode_batch(model, tokenizer, sentences, device, pooling = "cls"):
    input_ids = tokenizer(sentences, padding=True, max_length=512, truncation=True, return_tensors="pt",
                          add_special_tokens=True).to(device)
    features = model(**input_ids)[0]

    if pooling == "mean":
        features =  torch.sum(features[:,1:,:] * input_ids["attention_mask"][:,1:].unsqueeze(-1), dim=1) / torch.clamp(torch.sum(input_ids["attention_mask"][:,1:], dim=1, keepdims=True), min=1e-9)
    elif pooling == "cls":
        features = features[:,0,:]
    elif pooling == "mean+cls":
        mean_features = torch.sum(features[:,1:,:] * input_ids["attention_mask"][:,1:].unsqueeze(-1), dim=1) / torch.clamp(torch.sum(input_ids["attention_mask"][:,1:], dim=1, keepdims=True), min=1e-9)
        cls_features = features[:,0,:]
        features = torch.cat([mean_features, cls_features], dim=1)        
    return features

@st.cache(allow_output_mutation=True)
def load_sents_and_mpnet_vecs(load_pubmed=False):
    
    X = np.load("X_mpnet_bitfit:False_wiki_original_{}_v4.npy".format('mean'))
    #X = np.load("X_mpnet_bitfit:False_wiki_original_{}_v2.npy".format('mean'))
    # normalize 
    #X = X / np.linalg.norm(X, axis=1, keepdims=True)
    # with open("wiki_sents_10m.txt", "r") as f:
    #     lines = f.readlines()
    # lines = [l.strip() for l in lines]
    # lines = [s for s in lines if len(s.split()) > 5]
    # lines = lines[:5000000]


    # with open("wiki_sents_5m_v2.txt", "r") as f:
    #     lines = f.readlines()
    #     lines = [l.strip() for l in lines]

    with open("wiki_sents_10m_v2.txt", "r") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]

    if load_pubmed:
        with open("pubmed.txt", "r") as f:
            pubmed_data = f.readlines()
        pubmed_data = [s.strip() for s in pubmed_data]
        lines_pubmed = pubmed_data[:5000000]
    
        lines = lines + lines_pubmed
        #X_pubmed = np.load("X_mpnet_bitfit:False_pubmed_original_mean.npy")
        # concat
        #X  = np.concatenate([X, X_pubmed], axis=0)
    # normalize 
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    #X = X[:5000000]
    #lines = lines[:5000000]
    #good_idx = np.array([i for i in range(len(lines)) if len(lines[i].split(" ")) > 5 and len(lines[i].split(" ")) < 40])
    #X = X[good_idx]
    #lines = [lines[i] for i in good_idx]
    return X, lines


def fix_module_prefix_in_state_dict(state_dict):
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

@st.cache(allow_output_mutation=True)
def load_finetuned_model(finetuned=True,bitfit=False, model_name="mpnet"):
    with st.spinner('Loading BERT...'):
        if model_name == "roberta":
            sentence_encoder = AutoModel.from_pretrained("roberta-base")
            query_encoder = AutoModel.from_pretrained("roberta-base")
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        elif model_name == "mpnet":
            sentence_encoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            query_encoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

        prefix = "_mean" if METHOD == "mean" else "_cls"
        params_sent_encoder = torch.load("sentence_encoder5_mpnet_bitfit:False_final_mean_negatives:0.1_late:False_v2.pt")
        params_query_encoder = torch.load("query_encoder5_mpnet_bitfit:False_final_mean_negatives:0.1_late:False_v2.pt")
        params_linear_query = torch.load("linear_query5_mpnet_bitfit:False_final_mean_negatives:0.1_late:False_v2.pt")
        params_linear_sentence = torch.load("linear_sentence5_mpnet_bitfit:False_final_mean_negatives:0.1_late:False_v2.pt")
        params_sent_encoder = fix_module_prefix_in_state_dict(params_sent_encoder)
        params_query_encoder = fix_module_prefix_in_state_dict(params_query_encoder)
        params_linear_query = fix_module_prefix_in_state_dict(params_linear_query)
        params_linear_sentence = fix_module_prefix_in_state_dict(params_linear_sentence)


        sentence_encoder.load_state_dict(params_sent_encoder)
        query_encoder.load_state_dict(params_query_encoder)
        linear_query = torch.nn.Linear(768, 768)
        linear_query.load_state_dict(params_linear_query)
        linear_sentence = torch.nn.Linear(768, 768)
        linear_sentence.load_state_dict(params_linear_sentence)
        #sentence_encoder.eval()
        query_encoder.eval()

        return query_encoder, linear_query, linear_sentence, tokenizer

@st.cache(allow_output_mutation=True)
def load_pretrained_mpnet():

    with st.spinner('Loading BERT...'):
        mpnet = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        mpnet.eval()
        return mpnet, tokenizer

@st.cache(allow_output_mutation=True)
def load_finetuned_sentence_representations(load_pubmed=True):
        #X = np.load("X_mpnet_bitfit:False_wiki_and_pubmed_mean_no-negatives_v2.npy")
        X = np.load("X_mpnet_bitfit:False_wiki_mean_v5.npy")
        X = X[:10000000]
        # normalize
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X

def get_closest_neighbor_to_vector_by_cosine_sim(vec, X, sents, k=10):
    # get the closest neighbor for a given vector
    # normalize 
    vec = vec / np.linalg.norm(vec)
    #X = X / np.linalg.norm(X, axis=1, keepdims=True)
    sims = np.dot(X, vec)
    idx = np.argsort(sims)[::-1]
    return [sents[i] for i in idx[:k]]

device = "cuda:2"
pretrained_sentence_reps, sentences = load_sents_and_mpnet_vecs()
query_encoder, linear_query, linear_sentence, tokenizer = load_finetuned_model()
mpnet, tokenizer_pretrained = load_pretrained_mpnet()
finetuned_sentence_reps = load_finetuned_sentence_representations()

# use streamlit option for several default text inputs
query = st.text_input('Enter a characterization of the sentence', 'A unique measure of time.')

st.write("Or, choose one the following queries:")
queries = ["None", "a company which is a part of a larger company.", "a territory claimed by a country.", "A social faux pas committed by one person to another.", "a social faux pas comitted by a leader.", "monitoring of patient vital signs before treatment",
 "A book that influenced the development of a genre.", "A statistic from the U.S. Census Bureau.", "A project being realized due to a change in the political atmosphere",
"A man's attempt to gain influence over a monarch.", "A company leaving a specific industry.", "a person's upbringing in a particular city.",
"The act of teaching someone something.", "An investigation of a particular cell type revealed a substantial number of genes with altered expression.",
"A change in political ideology preventing a project from being completed.",
"The economic costs of a disease.", "An external factor influencing gene expression.", "Early detection of a certain health concern leads to a better treatment.",
"A specific compound is produced through a series of chemical reactions.", "Despite improvements in treatment, the long-term outcome remains below a desired threshold"]

# make a selectbox with the possible queries, with a default value
query_selectbox = st.selectbox('', queries)
if query_selectbox != "None":
    query = query_selectbox
query = query.lower()
query = "<query>: " + query

#model_type = st.selectbox('model', ('finetuned_mpnet', 'pretrained mpnet'))

# with torch.no_grad():

#     if model_type == "finetuned_mpnet":
#         query_rep = encode_cls(query_encoder, tokenizer, query, "cpu")
#         query_rep = linear_query(query_rep)[0]
#         sentence_reps = finetuned_sentence_reps
#     else:
#         query_rep = encode(mpnet, tokenizer, query, "cpu")[0]
#         sentence_reps = pretrained_sentence_reps
#     start = st.button('Run')
#     if start:
#         neighbors = get_closest_neighbor_to_vector_by_cosine_sim(query_rep, sentence_reps, sentences, k=100)


#         for n in neighbors:
#             st.write(n)

start = st.button('Run')


with torch.no_grad():

    #query_rep = encode_cls(query_encoder, tokenizer, query, "cpu")
    query_rep = encode_batch(query_encoder, tokenizer, [query], "cpu", "mean")[0]
    print("1", query_rep.shape)
    #query_rep = linear_query(query_rep)
    print("2", query_rep.shape)
    #query_rep = linear_query(query_rep)[0]
    sentence_reps = finetuned_sentence_reps
    #query_rep_orig = encode(mpnet, tokenizer, query.replace("<query>: ", ""), "cpu")[0]
    query_rep_orig = encode_batch(mpnet, tokenizer_pretrained, [query.replace("<query>: ", "")], "cpu", "cls")[0]
    sentence_reps_orig = pretrained_sentence_reps
    print("Running the query {}".format(query))
    print(sentence_reps.shape, sentence_reps_orig.shape)

    # print the norm of the difference vector between the weight of the linear_query and unit matrix
    print(np.linalg.norm(linear_query.weight.detach().cpu().numpy() - np.eye(768)))

    LATE=False
    if start:
        col1, col2 = st.columns(2)
        if LATE:
                cls_part = sentence_reps[:,768:]
                cls_part = cls_part * query_rep.detach().cpu().numpy()
                sentence_features = np.concatenate([sentence_reps[:,:768], cls_part], axis=1)
                sentence_features = sentence_features.reshape(sentence_features.shape[0], 2, 768)
                sentence_features = np.max(sentence_features, axis=1)
                query_rep = linear_query(torch.tensor(query_rep))[0].detach().cpu().numpy()
                sentence_features = linear_sentence(torch.tensor(sentence_features)).detach().cpu().numpy()
                neighbors = get_closest_neighbor_to_vector_by_cosine_sim(query_rep, sentence_features, sentences, k=1000)
    
        else:
            print(query_rep.shape, sentence_reps.shape)
            neighbors = get_closest_neighbor_to_vector_by_cosine_sim(query_rep, sentence_reps, sentences, k=1000)
        neighbors_orig = get_closest_neighbor_to_vector_by_cosine_sim(query_rep_orig, sentence_reps_orig, sentences, k=1000)
        col1.header("Ours")
        col2.header("Baseline (mpnet-v2)")

        for i, (n1,n2) in enumerate(zip(neighbors, neighbors_orig)):
            col1.write(n1)
            col2.write(n2)
        
