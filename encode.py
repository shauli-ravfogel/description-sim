import numpy as np
# import pca
from sklearn.decomposition import PCA
from datasets import load_dataset
import tqdm
import torch
from transformers import AutoModel, AutoTokenizer, ElectraForPreTraining, ElectraTokenizerFast
import random
import spacy
import numpy as np
from info_nce import info_nce_loss
import pickle


def encode_batch(model, tokenizer, sentences, device, pooling = "mean"):
    input_ids = tokenizer(sentences, padding=True, max_length=512, truncation=True, return_tensors="pt",
                          add_special_tokens=True).to(device)
    features = model(**input_ids)[0]

    if pooling == "mean":
        features = torch.sum(features[:,1:,:] * input_ids["attention_mask"][:,1:].unsqueeze(-1), dim=1) / torch.clamp(torch.sum(input_ids["attention_mask"][:,1:], dim=1, keepdims=True), min=1e-9)
    elif pooling == "cls":
        features = features[:,0,:]
    elif pooling == "mean+cls":
        # concatenate the mean and cls features. when calcualting the mean, ignore the cls token.
        mean_features = torch.sum(features[:,1:,:] * input_ids["attention_mask"][:,1:].unsqueeze(-1), dim=1) / torch.clamp(torch.sum(input_ids["attention_mask"][:,1:], dim=1, keepdims=True), min=1e-9)
        cls_features = features[:,0,:]
        features = torch.cat([mean_features, cls_features], dim=1)
    return features


def get_closest_neighbor_to_vector_by_cosine_sim(vec, X, sents, k=10):
    # get the closest neighbor for a given vector
    # normalize 
    vec = vec / np.linalg.norm(vec)
    assert (np.linalg.norm(X[0])-1)**2 < 1e-5
    
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    sims = np.dot(X, vec)
    idx = np.argsort(sims)[::-1]

    return [sents[i] for i in idx[:k]]


def get_rank_in_neighbors(vecs, X, sent_inds):
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    assert (np.linalg.norm(X[0])-1)**2 < 1e-5
    #X = X / np.linalg.norm(X, axis=1, keepdims=True)
    sims = vecs.dot(X.T)
    idx = np.argsort(sims, axis=1)[:,::-1]
    # return the rank of each sent_ind within idx
    ranks = []
    idx_lst = idx.tolist() 
    for i in range(len(sent_inds)):
        ranks.append(idx_lst[i].index(sent_inds[i]))
    return ranks

def get_closest_neighbor_to_vector_by_euclidean_distance(vec, X, sents, k=10):
    # get the closest neighbor for a given vector
    # normalize 
    sims = np.linalg.norm(X - vec, axis=1)
    idx = np.argsort(sims)

    return [sents[i] for i in idx[:k]]


# main

if __name__ == '__main__':

    bitfit=False
    model_type = "mpnet"
    pooling = "mean"
    include_pubmed=False
    pretrained=True

    if model_type == "roberta":
        sentence_encoder = AutoModel.from_pretrained("roberta-base")
        query_encoder = AutoModel.from_pretrained("roberta-base")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    elif model_type == "mpnet":
        sentence_encoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        query_encoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentence_encoder= torch.nn.DataParallel(sentence_encoder)
    query_encoder= torch.nn.DataParallel(query_encoder)
    sentence_encoder.to(device)
    query_encoder.to(device)

    if pretrained:
        # load mdoel parameters
        linear_sentence = torch.nn.Linear(768, 768)
        sentence_encoder_dict = torch.load("sentence_encoder5_mpnet_bitfit:False_final_mean_negatives:0.1_late:False_v2.pt")
        #sentence_encoder_dict = fix_module_prefix_in_state_dict_electra(sentence_encoder_dict)
        sentence_encoder.load_state_dict(sentence_encoder_dict)
        #query_encoder_dict = torch.load("query_encoder3_{}_bitfit:{}_final_mean_no-negatives_v2.pt".format(model_type,False, pooling))
        #query_encoder_dict = fix_module_prefix_in_state_dict_electra(query_encoder_dict)
        #query_encoder.load_state_dict(query_encoder_dict)
        #linear_query_dict = torch.load("linear_query3_{}_bitfit:{}_final_cls_v2.pt".format(model_type,False))
        #linear_query_dict = fix_module_prefix_in_state_dict_electra(linear_query_dict)
        #linear_query.load_state_dict(linear_query_dict)
        linear_sentence_dict = torch.load("linear_sentence5_mpnet_bitfit:False_final_mean_negatives:0.1_late:False_v2.pt")
        #linear_sentence_dict = fix_module_prefix_in_state_dict_electra(linear_sentence_dict)
        linear_sentence.load_state_dict(linear_sentence_dict)


    # encode all sentences in batches.
    import tqdm
    batch_size = 512
    X = []
    sents = []
    sentence_encoder.to(device)
    #linear_sentence.to(device)

    # with open("wiki_sents_5m_v2.txt", "r") as f:
    #     lines = f.readlines()
    #     lines = [l.strip() for l in lines]
    #     #lines = [s for s in lines if len(s.split()) > 5]
    # with open("wiki_sents_5m_part2_v2.txt", "r") as f:
    #     lines2 = f.readlines()
    #     lines2 = [l.strip() for l in lines2]
    
    # lines = list(set(lines + lines2))

    with open("wiki_sents_10m_v2.txt", "r") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]

    with open("pubmed.txt", "r") as f:
        pubmed_data = f.readlines()
    pubmed_data = [s.strip() for s in pubmed_data]
    if include_pubmed:
        lines = lines + pubmed_data[:5000000]

    #lines = pubmed_data[:5000000]
    for i in tqdm.tqdm(range(0, len(lines), batch_size)):
        batch = [d for d in lines[i:i+batch_size]]
        sents += batch
        with torch.no_grad():
            h = encode_batch(sentence_encoder, tokenizer, batch, device, pooling=pooling).detach().cpu().numpy()
            #h = linear_sentence(torch.from_numpy(h).to(device)).detach().cpu().numpy()
            X.append(h)

    

    X = np.concatenate(X, axis=0)
    if include_pubmed:
        np.save("X_{}_bitfit:{}_wiki_and_pubmed_mean_v5.npy".format(model_type, bitfit), X)
    else:
        if pretrained:
            np.save("X_{}_bitfit:{}_wiki_mean_v5.npy".format(model_type, bitfit), X)
        else:
            np.save("X_{}_bitfit:{}_wiki_original_mean_v5.npy".format(model_type, bitfit), X)
