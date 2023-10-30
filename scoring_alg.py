import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils_SA import *
from utils_attack import *


def patient_code_score(model, continuous, patient, rxdx_codes, sorted_visits, s_real, increase=False):
    # code scores
    code_ds_dict = {}
    for visit in sorted_visits:
        code_indexes = find_nonzero_indexes(patient[visit, 2:])
        codes = zip(code_indexes, rxdx_codes[code_indexes])
        code_ds_dict_v = {}
        for index, code in codes:
            patient_copy2 = patient.copy()
            patient_copy2[visit, index + 2]=0
            code_score = pred_duration(model, continuous, patient_copy2).numpy()[0]
            if increase:
                code_ds = code_score - s_real
            else:
                code_ds = s_real - code_score
            code_ds_dict_v[index + 2] = code_ds
        code_ds_dict[visit] = sorted(code_ds_dict_v.items(), key=lambda x: x[1], reverse=True)

    return code_ds_dict

def patient_code_score2(model, continuous, patient, rxdx_codes, increase=False):
    """this function score all the codes inside the history of a patient
    search space is ONLY based on the scores of existing codes  =>  not optimized
    """

    s_real = pred_duration(model,continuous, patient).numpy()[0]
    code_ds_dict = {}
    num_visit = patient.shape[0]
    for visit in range(num_visit):
        code_indexes = find_nonzero_indexes(patient[visit, 2:])
        codes = zip(code_indexes, rxdx_codes[code_indexes])
        #code_ds_dict_v = {}
        for index, code in codes:
            patient_copy2 = patient.copy()
            patient_copy2[visit, index + 2]=0
            code_score = pred_duration(model, continuous, patient_copy2).numpy()[0]
            if increase:
                code_ds = code_score - s_real
            else:
                code_ds = s_real - code_score
            code_ds_dict[(visit, index + 2)] = code_ds
    sorted_codes = sorted(code_ds_dict.items(), key=lambda x: x[1], reverse=True)

    return code_ds_dict, sorted_codes, s_real


def patient_code_score_new(model,continuous, patient, rxdx_codes, onto_rx, onto_dx,
                           code_ctoi, co_occurrence_prob_matrix, p_cooccur, increase=False):
    """this function score all the {{codes inside the history of a patient} + {potential codes to be added}}
    {potential codes to be added}} synonym codes of existing codes that
        are not already in the existing codes
        have co-occurences of more than p with their origin codes
    """
    #scoring the existing codes + finding the synonyms
    s_real = pred_duration(model, continuous, patient).numpy()[0]
    code_ds_dict = {}
    num_visit = patient.shape[0]
    for visit in range(num_visit):
        code_indexes = find_nonzero_indexes(patient[visit, 2:])
        codes = zip(code_indexes, rxdx_codes[code_indexes])
        for code_idx, code in codes:

            if code_idx+2 >= 1126:
                synonym_set = get_synonym_codes(code_idx, True, onto_rx, rxdx_codes)
            else:
                synonym_set = get_synonym_codes(code_idx, False, onto_dx, rxdx_codes)

            siblings_indices = [code_ctoi[sibling] for sibling in synonym_set if sibling in rxdx_codes]

            if len(siblings_indices)!=0:
                sibilings_cooccurence_indices, sibilings_cooccurence_probs = get_sort_cooccurence(code_idx+2, list(np.array(siblings_indices)+2),co_occurrence_prob_matrix)

                highly_cooccured_nonexistent_siblings = [sib_idx for e,sib_idx in enumerate(sibilings_cooccurence_indices) if (sibilings_cooccurence_probs[e] >= p_cooccur) & (patient[visit, sib_idx]!=1)]

                for co_code_idx in highly_cooccured_nonexistent_siblings:
                    patient_copy1 = patient.copy()
                    patient_copy1[visit, co_code_idx]=1
                    code_score = pred_duration(model, continuous, patient_copy1).numpy()[0]
                    if increase:
                        code_ds = code_score - s_real
                    else:
                        code_ds = s_real - code_score
                    code_ds_dict[(visit, 'add', co_code_idx, code_idx+2)] = code_ds



            patient_copy2 = patient.copy()
            patient_copy2[visit, code_idx + 2] = 0
            code_score = pred_duration(model, continuous, patient_copy2).numpy()[0]
            if increase:
                code_ds = code_score - s_real
            else:
                code_ds = s_real - code_score
            code_ds_dict[(visit, 'remove', code_idx + 2, tuple(synonym_set))] = code_ds

    sorted_codes = sorted(code_ds_dict.items(), key=lambda x: x[1], reverse=True)

    return code_ds_dict, sorted_codes, s_real

def patient_code_score_new_ScoreSym1_gpu(model, encoder_gpu, patient, rxdx_codes, onto_rx, onto_dx,
                           code_ctoi, co_occurrence_prob_matrix, p_cooccur, dx_names, rx_names, ctoi_dx, ctoi_rx, device, increase=False):

    #scoring the existing codes + finding the synonyms
    patient_versons = []
    patient_versons.append(patient[np.newaxis,:,:])
    s_real = pred_duration(model, patient).numpy()[0]
    code_ds_dict = {}
    num_visit = patient.shape[0]
    for visit in range(num_visit):
        code_indexes = find_nonzero_indexes(patient[visit, 2:])
        codes = zip(code_indexes, rxdx_codes[code_indexes])

        for code_idx, code in codes:
            #chosen_rep_id=0
            if code_idx+2 >= 1126:
                synonym_set = get_synonym_codes(code_idx, True, onto_rx, rxdx_codes)
            else:
                synonym_set = get_synonym_codes(code_idx, False, onto_dx, rxdx_codes)

            siblings_indices = [code_ctoi[sibling] for sibling in synonym_set if sibling in rxdx_codes]
            #siblings_indices ---> includes first 2 indices

            if len(siblings_indices)!=0:
                sibilings_cooccurence_indices, sibilings_cooccurence_probs = get_sort_cooccurence(code_idx+2, list(np.array(siblings_indices)+2),co_occurrence_prob_matrix)
            #siblings_indices includes first 2 indices

                highly_cooccured_nonexistent_siblings = [sib_idx for e,sib_idx in enumerate(sibilings_cooccurence_indices) if (sibilings_cooccurence_probs[e] >= p_cooccur) & (patient[visit, sib_idx]!=1)] #upper than the threshold and nonexistant

                #maximum = 0
                #chosen_rep_id = 0
                for co_code_idx in highly_cooccured_nonexistent_siblings:
                    patient_copy1 = patient.copy()
                    patient_copy1[visit, co_code_idx]=1
                    patient_versons.append(patient_copy1[np.newaxis,:,:])
                    code_score = pred_duration(model, patient_copy1).numpy()[0]
                    #simlarity = get_similarity(patient, patient_copy1 , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                    if increase:
                        code_ds = code_score - s_real
                    else:
                        code_ds = s_real - code_score
                    code_ds_dict[(visit, 'add', co_code_idx, code_idx+2)] = code_ds #* sym_weight_func(simlarity)

            patient_copy2 = patient.copy()
            patient_copy2[visit, code_idx + 2] = 0
            patient_versons.append(patient_copy2[np.newaxis,:,:])
            #simlarity1 = get_similarity(patient, patient_copy2 , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
            code_score = pred_duration(model, patient_copy2).numpy()[0]
            if increase:
                code_ds = code_score - s_real
            else:
                code_ds = s_real - code_score
            code_ds_dict[(visit, 'remove', code_idx + 2, tuple(synonym_set))] = code_ds #* sym_weight_func(max(simlarity1, simlarity2))


    score_list = get_similarity2(patient_versons, encoder_gpu,
                   onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx, device)

    weighetd_code_ds_dict = {key: value * sym_weight_func(score_list[e]) for e , (key, value) in enumerate(code_ds_dict.items())}

    sorted_codes = sorted(weighetd_code_ds_dict.items(), key=lambda x: x[1], reverse=True)
    #sorted_codes = sorted(code_ds_dict.items(), key=lambda x: x[1], reverse=True)

    return code_ds_dict, sorted_codes, s_real

def patient_code_score_new_ScoreSym1_gpu2(model, encoder_gpu, patient, rxdx_codes, onto_rx, onto_dx,
                           code_ctoi, co_occurrence_prob_matrix, p_cooccur, dx_names, rx_names, ctoi_dx, ctoi_rx, device, increase=False):

    #scoring the existing codes + finding the synonyms
    patient_versons = []
    patient_versons.append(patient[np.newaxis,:,:])
    s_real = pred_duration(model, patient).numpy()[0]
    code_ds_dict = {}
    num_visit = patient.shape[0]
    for visit in range(num_visit):
        code_indexes = find_nonzero_indexes(patient[visit, 2:])
        codes = zip(code_indexes, rxdx_codes[code_indexes])

        for code_idx, code in codes:
            #chosen_rep_id=0
            if code_idx+2 >= 1126:
                synonym_set = get_synonym_codes(code_idx, True, onto_rx, rxdx_codes)
            else:
                synonym_set = get_synonym_codes(code_idx, False, onto_dx, rxdx_codes)

            siblings_indices = [code_ctoi[sibling] for sibling in synonym_set if sibling in rxdx_codes]
            #siblings_indices ---> includes first 2 indices

            if len(siblings_indices)!=0:
                sibilings_cooccurence_indices, sibilings_cooccurence_probs = get_sort_cooccurence(code_idx+2, list(np.array(siblings_indices)+2),co_occurrence_prob_matrix)
            #siblings_indices includes first 2 indices

                highly_cooccured_nonexistent_siblings = [sib_idx for e,sib_idx in enumerate(sibilings_cooccurence_indices) if (sibilings_cooccurence_probs[e] >= p_cooccur) & (patient[visit, sib_idx]!=1)] #upper than the threshold and nonexistant

                #maximum = 0
                #chosen_rep_id = 0
                for co_code_idx in highly_cooccured_nonexistent_siblings:
                    patient_copy1 = patient.copy()
                    patient_copy1[visit, co_code_idx]=1
                    patient_versons.append(patient_copy1[np.newaxis,:,:])
                    code_score = pred_duration(model, patient_copy1).numpy()[0]
                    #simlarity = get_similarity(patient, patient_copy1 , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                    if increase:
                        code_ds = code_score - s_real
                    else:
                        code_ds = s_real - code_score
                    code_ds_dict[(visit, 'add', co_code_idx, code_idx+2)] = code_ds #* sym_weight_func(simlarity)

            patient_copy2 = patient.copy()
            patient_copy2[visit, code_idx + 2] = 0
            patient_versons.append(patient_copy2[np.newaxis,:,:])
            #simlarity1 = get_similarity(patient, patient_copy2 , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
            code_score = pred_duration(model, patient_copy2).numpy()[0]
            if increase:
                code_ds = code_score - s_real
            else:
                code_ds = s_real - code_score
            code_ds_dict[(visit, 'remove', code_idx + 2, tuple(synonym_set))] = code_ds #* sym_weight_func(max(simlarity1, simlarity2))


    score_list = get_similarity2(patient_versons, encoder_gpu,
                   onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx, device)

    weighetd_code_ds_dict = {key: value * sym_weight_func2(score_list[e]) for e , (key, value) in enumerate(code_ds_dict.items())}

    sorted_codes = sorted(weighetd_code_ds_dict.items(), key=lambda x: x[1], reverse=True)
    #sorted_codes = sorted(code_ds_dict.items(), key=lambda x: x[1], reverse=True)

    return code_ds_dict, sorted_codes, s_real

