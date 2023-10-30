import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils_SA import *

def pred_duration(model,continuous, x):
    """calculating the estimated mean lifetime for discrete-time survival analysis models"""
    if continuous:
        """calculating the estimated mean lifetime for continuous-time survival analysis models"""
        x = x.reshape(1,-1)
        surv_probs = model.predict_surv_df(x)
        pred_duration = []
        for i in range(surv_probs.shape[1]):
            area = np.trapz(surv_probs.values[:,i], x=surv_probs.index)
            pred_duration.append(area)
        pred_dur = torch.tensor(pred_duration)
    else:
        x = x.reshape(1,-1)
        surv_probs = torch.from_numpy(model.predict_surv(x))
        pred_dur = torch.sum(surv_probs, dim=-1)

    return pred_dur

def find_nonzero_indexes(arr):
    """finding non-zero indices of an array"""
    nonzero_indexes = [i for i, val in enumerate(arr) if val != 0]
    return nonzero_indexes

def patient_visit_score(model, continuous, patient, increase=False):
    num_visits = patient.shape[0]
    # visit scores
    ds_dict={}
    s_real = pred_duration(model,continuous, patient).numpy()[0]
    for i in range(num_visits):
        patient_copy = patient.copy()
        patient_copy[i]=0
        s = pred_duration(model,patient_copy).numpy()
        if increase:
            ds = s - s_real
        else:
            ds = s_real - s
        ds_dict[i]=ds

    sorted_visits, sorted_visit_scores = zip(*sorted(ds_dict.items(), key=lambda x:x[1], reverse=True))
    return sorted_visits, sorted_visit_scores, s_real


def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtracting max(x) for numerical stability
    return e_x / e_x.sum(axis=0)



def get_synonym_codes(code_idx, rx, onto_vocab, rxdx_codes):
    if rx==True:
        act3_class = onto_vocab[onto_vocab['atc4'] == rxdx_codes[code_idx]]['atc3'].values[0]
        synonym_set = onto_vocab[onto_vocab['atc3']==act3_class]['atc4'].unique()
    else:
        l2_class = onto_vocab[onto_vocab['l3'] == rxdx_codes[code_idx]]['l2'].values[0]
        synonym_set = onto_vocab[onto_vocab['l2'] == l2_class]['l3'].unique()

    synonym_set = synonym_set[synonym_set != rxdx_codes[code_idx]]
    return list(synonym_set)


def calculate_cooccurrence_matrix(patient_matrices):
    # Initialize the co-occurrence matrix with zeros
    num_feature = patient_matrices.shape[-1]
    cooccurrence_matrix = np.zeros((num_feature, num_feature), dtype='int32')

    # Iterate through the patient matrices
    for patient_matrix in tqdm(patient_matrices):
        # Calculate the co-occurrence for each pair of medical codes
        cooccurrence_matrix += patient_matrix.T.dot(patient_matrix)
    return cooccurrence_matrix / np.diag(cooccurrence_matrix)


def find_top_k_cooccurence_codes(query_code_index, co_prob_matrix, k):
    """finding the top k codes with the highest co-occurence with the given code"""
    top_k_cooccuring_codes_index, top_k_cooccuring_codes_probs = zip(*sorted(zip(np.arange(len(co_prob_matrix[query_code_index])) , co_prob_matrix[query_code_index]), key=lambda x:x[1], reverse=True)[:k])
    return top_k_cooccuring_codes_index[1:], top_k_cooccuring_codes_probs[1:]


def get_sort_cooccurence(query_code_index, siblings_indices, co_prob_matrix):
    """ order the siblings based on their cooccurence with the given code,
        returing the ordered codes + scores
    """
    sibilings_cooccurence_indices, sibilings_cooccurence_probs = zip(
        *sorted(zip(siblings_indices, co_prob_matrix[query_code_index - 2, list(np.array(siblings_indices) - 2)]),
                key=lambda x: x[1], reverse=True))
    #sibilings_cooccurence_indices -> include first 2 indice

    return sibilings_cooccurence_indices, sibilings_cooccurence_probs


def get_similarity(x_attacked_patient, adv_x_test1, encoder,
                   onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx):

    "calculating similarity"

    features = np.concatenate([x_attacked_patient[np.newaxis,:,:],
                               adv_x_test1[np.newaxis,:,:]], axis=0)

    x_dx, x_rx = convert_onehot_to_idx(features, onto_dx, onto_rx, dx_names, rx_names,
                                       ctoi_dx, ctoi_rx, max_len=125, pad_token=0)
    x_demo = 0

    encoder.eval()
    _, compressed_incoded_visit, _, _ = encoder(x_dx, x_rx, x_demo)

    compressed_incoded_visit = compressed_incoded_visit.detach().numpy()
    score1 = cosine_similarity(compressed_incoded_visit[0].reshape(1,-1),
                               compressed_incoded_visit[1].reshape(1,-1))

    return score1


def get_similarity2(patient_list, encoder,
                   onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx, device):
    "calculating similarity using GPU"

    features = np.concatenate(patient_list, axis=0)

    x_dx, x_rx = convert_onehot_to_idx(features, onto_dx, onto_rx, dx_names, rx_names,
                                       ctoi_dx, ctoi_rx, max_len=125, pad_token=0)
    x_demo = 0

    encoder.eval()
    x_dx = [i.to(device) for i in x_dx]
    x_rx = [i.to(device) for i in x_rx]
    #x_demo = x_demo.to(device)
    _, compressed_incoded_visit, _, _ = encoder(x_dx, x_rx, x_demo)

    compressed_incoded_visit = compressed_incoded_visit.detach().cpu().numpy()
    score_list=[]
    for i in range(1, len(compressed_incoded_visit)):
        score = cosine_similarity(compressed_incoded_visit[0].reshape(1,-1),
                               compressed_incoded_visit[i].reshape(1,-1))
        score_list.append(score)

    return score_list



