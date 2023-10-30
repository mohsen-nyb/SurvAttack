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


def sym_weight_func(sym):
    if sym>0.95:
        return np.exp(sym)
    elif (sym<=0.95)&(sym>0.9):
        return np.exp(0.5*sym)
    elif (sym<=0.9)&(sym>0.8):
        return np.exp(-0.5*sym)
    else:
        return np.exp(-2*sym)

def sym_weight_func2(sym):
    if sym> 0.99:
        return np.exp(3*sym)
    elif (sym<=0.99)&(sym>0.98):
        return np.exp(2*sym)
    elif (sym<=0.98)&(sym>0.95):
        return np.exp(1.0*sym)
    elif (sym<=0.95)&(sym>0.9):
        return np.exp(0.5*sym)
    elif (sym<=0.9)&(sym>0.8):
        return np.exp(-0.5*sym)
    else:
        return np.exp(-2*sym)

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


def Bbox_AdvAttack_SA_pair_new(model, patient1, patient2, rxdx_codes, onto_rx, onto_dx, co_occurrence_prob_matrix, cooccrence_ratio=0.15, increase=False, verbose=True):

    """flip the predicted time/risk of two patients based on only code score
    removing + replacing + adding
    search space is based on the scores of both existing codes and potential adding codes
    """

    if not increase:
        x_fixed_patient, x_attacked_patient = patient1, patient2
    else:
        x_attacked_patient, x_fixed_patient = patient1, patient2

    code_ctoi = {code:i for i, code in enumerate(rxdx_codes)}  #includes first 2 indices
    comparing_score = pred_duration(model, x_fixed_patient).numpy()[0]


    #the new sorting consider both
    code_ds_dict, sorted_codes, s_real = patient_code_score_new(model,
                                                                x_attacked_patient,
                                                            rxdx_codes[2:],
                                                                onto_rx,
                                                                onto_dx,
                                                                code_ctoi,
                                                                co_occurrence_prob_matrix,
                                                                p_cooccur=cooccrence_ratio,
                                                                increase=increase)
    num_visits = x_attacked_patient.shape[0]

    if (not increase) & (s_real < comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, None, comparing_score, s_real

    if (increase) & (s_real > comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, None, comparing_score, s_real

    adv_x_test = x_attacked_patient.copy()
    removed_codes = []
    replaced_codes = []
    added_codes = []
    final_adv = 0

    if not increase:
        for item in sorted_codes:

            visit_idx = item[0][0]
            label = item[0][1]
            code_idx = item[0][2]

            current_score = pred_duration(model, adv_x_test).numpy()[0]

            if label == 'remove':
                if adv_x_test[visit_idx, code_idx]==1:
                    adv_x_test[visit_idx, code_idx]=0
                    score_removed = pred_duration(model, adv_x_test).numpy()[0]

                    if score_removed < min(current_score, s_real):
                        removed_codes.append((visit_idx, code_idx))

                        #check attack
                        if score_removed < comparing_score:
                            final_adv = adv_x_test
                            break
                    else:
                        adv_x_test[visit_idx, code_idx]=1


                 # checking replacing
                #if code_idx >= 1126:
                synonym_set = item[0][3]
                #synonym_set = get_synonym_codes(code_idx, True, onto_vocab_rx, rxdx_codes)
                ##################### replacing
                chosen_syn=0
                maximum=0
                for synonym_code in synonym_set:
                    if synonym_code in rxdx_codes:
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx]=0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp < current_score) & (np.abs(s_temp-current_score) > maximum):
                            maximum = np.abs(s_temp-current_score)
                            chosen_syn = synonym_code

                if chosen_syn!=0:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn]]==0:
                        if adv_x_test[visit_idx, code_idx]==0:
                            already_zero=1
                        else:
                            already_zero=0

                        adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=1
                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score < min(s_real, current_score, score_removed):
                            replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                            if already_zero==0:
                                removed_codes.append((visit_idx, code_idx))
                            if added_syn_score < comparing_score:
                                final_adv = adv_x_test
                                break
                        elif (added_syn_score < min(s_real, current_score)) & (added_syn_score > score_removed):
                            adv_x_test[visit_idx, code_idx]=0
                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                        else:
                            if already_zero==1:
                                adv_x_test[visit_idx, code_idx]=0
                            else:
                                adv_x_test[visit_idx, code_idx]=1

                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0

            else: #label=='add'
                origin = item[0][3]
                if adv_x_test[visit_idx, code_idx]==0:
                    adv_x_test[visit_idx, code_idx]=1
                    score_added = pred_duration(model, adv_x_test).numpy()[0]

                    if score_added < min(current_score, s_real):
                        added_codes.append((visit_idx, origin, code_idx))


                        #check attack
                        if score_added < comparing_score:
                            final_adv = adv_x_test
                            break
                    else:
                        adv_x_test[visit_idx, code_idx]=0



        if type(final_adv) == int:
            print('attack failed')
            final_score = pred_duration(model, adv_x_test).numpy()[0]
            change_rate = 100 * (len(added_codes) + len(removed_codes)) / x_attacked_patient[:, 2:].sum()
        else:
            final_score = pred_duration(model, final_adv).numpy()[0]
            change_rate = 100 * (len(added_codes) + len(removed_codes)) / x_attacked_patient[:, 2:].sum()
            if verbose:
                print('Black-Box attack is done => patient pred-time orders has been attacked!')
                print(f'original pred-time orders: p1 :{comparing_score:.3f} < p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} > p2 :{final_score:.3f}')
                print(f'number of perturbations: {len(removed_codes) + len(added_codes)}')
                print(f'number of removed : {len(removed_codes) - len(replaced_codes)}')
                print(f'number of replaced: {len(replaced_codes)}')
                print(f'number of added: {len(added_codes)}')
                print(f'change rate: {change_rate:.2f}%')
                visit_of_change = np.array([item[0] for item in removed_codes] + [item[0] for item in added_codes])
                for i in range(num_visits):
                    print(f'number of changes in visit {i + 1} : {(visit_of_change == i).sum()} / {x_attacked_patient[i, 2:].sum()}')

    #increase
    else:
        for item in sorted_codes:

            visit_idx = item[0][0]
            label = item[0][1]
            code_idx = item[0][2]

            current_score = pred_duration(model, adv_x_test).numpy()[0]

            if label == 'remove':
                if adv_x_test[visit_idx, code_idx]==1:
                    adv_x_test[visit_idx, code_idx]=0
                    score_removed = pred_duration(model, adv_x_test).numpy()[0]

                    if score_removed > min(current_score, s_real):
                        removed_codes.append((visit_idx, code_idx))

                        #check attack
                        if score_removed > comparing_score:
                            final_adv = adv_x_test
                            break
                    else:
                        adv_x_test[visit_idx, code_idx]=1


                 # checking replacing
                #if code_idx >= 1126:
                synonym_set = item[0][3]
                #synonym_set = get_synonym_codes(code_idx, True, onto_vocab_rx, rxdx_codes)
                ##################### replacing
                chosen_syn=0
                maximum=0
                for synonym_code in synonym_set:
                    if synonym_code in rxdx_codes:
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx]=0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp > current_score) & (np.abs(s_temp-current_score) > maximum):
                            maximum = np.abs(s_temp-current_score)
                            chosen_syn = synonym_code

                if chosen_syn!=0:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn]]==0:
                        if adv_x_test[visit_idx, code_idx]==0:
                            already_zero=1
                        else:
                            already_zero=0

                        adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=1
                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score > max(s_real, current_score, score_removed):
                            replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                            if already_zero==0:
                                removed_codes.append((visit_idx, code_idx))
                            if added_syn_score > comparing_score:
                                final_adv = adv_x_test
                                break
                        elif (added_syn_score > max(s_real, current_score)) & (added_syn_score < score_removed):
                            adv_x_test[visit_idx, code_idx]=0
                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                        else:
                            if already_zero==1:
                                adv_x_test[visit_idx, code_idx]=0
                            else:
                                adv_x_test[visit_idx, code_idx]=1

                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0

            else: #label=='add'
                origin = item[0][3]
                if adv_x_test[visit_idx, code_idx]==0:
                    adv_x_test[visit_idx, code_idx]=1
                    score_added = pred_duration(model, adv_x_test).numpy()[0]

                    if score_added > max(current_score, s_real):
                        added_codes.append((visit_idx, origin, code_idx))


                        #check attack
                        if score_added > comparing_score:
                            final_adv = adv_x_test
                            break
                    else:
                        adv_x_test[visit_idx, code_idx]=0



        if type(final_adv) == int:
            print('attack failed')
            final_score = pred_duration(model, adv_x_test).numpy()[0]
            change_rate = 100 * (len(added_codes) + len(removed_codes)) / x_attacked_patient[:, 2:].sum()
        else:
            final_score = pred_duration(model, final_adv).numpy()[0]
            change_rate = 100 * (len(added_codes) + len(removed_codes)) / x_attacked_patient[:, 2:].sum()
            if verbose:
                print('Black-Box attack is done => patient pred-time orders has been attacked!')
                print(f'original pred-time orders: p1 :{comparing_score:.3f} > p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} < p2 :{final_score:.3f}')
                print(f'number of perturbations: {len(removed_codes) + len(added_codes)}')
                print(f'number of removed : {len(removed_codes) - len(replaced_codes)}')
                print(f'number of replaced: {len(replaced_codes)}')
                print(f'number of added: {len(added_codes)}')
                print(f'change rate: {change_rate:.2f}%')
                visit_of_change = np.array([item[0] for item in removed_codes] + [item[0] for item in added_codes])
                for i in range(num_visits):
                    print(f'number of changes in visit {i + 1} : {(visit_of_change == i).sum()} / {x_attacked_patient[i, 2:].sum()}')

    return final_adv, removed_codes, replaced_codes, added_codes, final_score, change_rate, comparing_score, s_real

def Bbox_AdvAttack_SA_pair(model, patient1, patient2, rxdx_codes, onto_rx, onto_dx, co_occurrence_prob_matrix, cooccrence_ratio=0.15, increase=False, verbose=True):
    """flip the predicted time/risk of two patients based on only code score"""

    if not increase:
        x_fixed_patient, x_attacked_patient = patient1, patient2
    else:
        x_attacked_patient, x_fixed_patient = patient1, patient2

    code_itoc = {i:code for i, code in enumerate(rxdx_codes)}
    code_ctoi = {code:i for i, code in enumerate(rxdx_codes)}
    comparing_score = pred_duration(model, x_fixed_patient).numpy()[0]

    code_ds_dict, sorted_codes, s_real = patient_code_score2(model, x_attacked_patient, rxdx_codes[2:], increase=increase)

    num_visits = x_attacked_patient.shape[0]

    if (not increase) & (s_real < comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, comparing_score, s_real

    if (increase) & (s_real > comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, comparing_score, s_real

    adv_x_test = x_attacked_patient.copy()
    removed_codes = []
    replaced_codes = []
    added_codes = []
    final_adv = 0

    if not increase:
        for item in sorted_codes:
            removed = False
            visit_idx = item[0][0]
            code_idx = item[0][1]
            current_score = pred_duration(model, adv_x_test).numpy()[0]

            adv_x_test[visit_idx, code_idx]=0
            score = pred_duration(model, adv_x_test).numpy()[0]

            if score < min(current_score, s_real):
                removed_codes.append((visit_idx, code_idx))
                removed=True

                if score < comparing_score:
                    final_adv = adv_x_test
                    break

            else:
                adv_x_test[visit_idx, code_idx]=1

             #checking replacing
            if code_idx >= 1126:
                replacing = False
                synonym_set = get_synonym_codes(code_idx, True, onto_rx, rxdx_codes)

                ######################################### replacing
                chosen_syn=0
                maximum = 0
                for synonym_code in synonym_set:
                    if synonym_code in rxdx_codes:
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx]=0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp < current_score) & (np.abs(s_temp-current_score) > maximum):
                            maximum = np.abs(s_temp-current_score)
                            chosen_syn = synonym_code

                if chosen_syn!=0:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn]]==0:
                        if adv_x_test[visit_idx, code_idx]==0:
                            already_zero=1
                        else:
                            already_zero=0

                        adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=1
                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score < min(s_real, current_score, score):
                            replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                            replacing = True
                            if already_zero==0:
                                removed = True
                                removed_codes.append((visit_idx, code_idx))
                            if added_syn_score < comparing_score:
                                final_adv = adv_x_test
                                break
                        elif (added_syn_score < min(s_real, current_score)) & (added_syn_score > score):
                            adv_x_test[visit_idx, code_idx]=0
                            removed = True
                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                        else:
                            if already_zero==1:
                                adv_x_test[visit_idx, code_idx]=0
                            else:
                                adv_x_test[visit_idx, code_idx]=1

                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0


                #########################################adding
                synonym_set_copy=synonym_set.copy()
                if replacing:
                    synonym_set_copy.remove(chosen_syn)
                siblings_indices = [code_ctoi[sibling] for sibling in synonym_set_copy if sibling in rxdx_codes]
                if len(siblings_indices)!=0:
                    sibilings_cooccurence_indices, sibilings_cooccurence_probs = get_sort_cooccurence(code_idx, siblings_indices, co_occurrence_prob_matrix)

                    mx_scr=0
                    mx_added_sib_idx=0
                    for e, cooccured_sib_idx in enumerate(sibilings_cooccurence_indices):
                        if sibilings_cooccurence_probs[e] >= cooccrence_ratio:
                            if adv_x_test[visit_idx, cooccured_sib_idx]==0:
                                adv_x_test_added_copy = adv_x_test.copy()
                                adv_x_test_added_copy[visit_idx, cooccured_sib_idx]=1
                                test_added_scr = pred_duration(model, adv_x_test_added_copy).numpy()[0]
                                if replacing:
                                    if (test_added_scr < min(current_score, added_syn_score)) & (np.abs(test_added_scr - added_syn_score) > mx_scr):
                                        mx_scr = test_added_scr
                                        mx_added_sib_idx = cooccured_sib_idx

                                else:
                                    min_score = min(current_score, score)
                                    if (test_added_scr < min_score) & (np.abs(test_added_scr - min_score) > mx_scr):
                                        mx_scr = test_added_scr
                                        mx_added_sib_idx = cooccured_sib_idx

                    if mx_added_sib_idx!=0:
                        adv_x_test[visit_idx, mx_added_sib_idx]=1
                        added_codes.append((visit_idx, code_idx, mx_added_sib_idx))
                        if mx_scr < comparing_score:
                            final_adv = adv_x_test
                            break


            else:
                replacing = False
                synonym_set_dx = get_synonym_codes(code_idx, False, onto_dx, rxdx_codes)

                #########################################replacing
                chosen_syn_dx=0
                maximum = 0
                for synonym_code in synonym_set_dx:
                    if synonym_code in rxdx_codes:
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx]=0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp < current_score) & (np.abs(s_temp-current_score) > maximum):
                            maximum = np.abs(s_temp-current_score)
                            chosen_syn_dx = synonym_code

                if chosen_syn_dx!=0:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]]==0:
                        if adv_x_test[visit_idx, code_idx]==0:
                            already_zero=1
                        else:
                            already_zero=0
                        adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]]=1
                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score < min(s_real, current_score, score):
                            replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn_dx]))
                            replacing = True
                            if already_zero==0:
                                removed_codes.append((visit_idx, code_idx))
                                removed=True

                            if added_syn_score < comparing_score:
                                final_adv = adv_x_test
                                break

                        elif (added_syn_score < min(s_real, current_score)) & (added_syn_score > score):
                            adv_x_test[visit_idx, code_idx]=0
                            adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]]=0
                        else:
                            if already_zero==1:
                                adv_x_test[visit_idx, code_idx]=0
                            else:
                                adv_x_test[visit_idx, code_idx]=1
                            adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]]=0


                ########################################  adding
                synonym_set_copy = synonym_set_dx.copy()
                if replacing:
                    synonym_set_copy.remove(chosen_syn_dx)
                siblings_indices = [code_ctoi[sibling] for sibling in synonym_set_copy if sibling in rxdx_codes]
                if len(siblings_indices)!=0:
                    sibilings_cooccurence_indices, sibilings_cooccurence_probs = get_sort_cooccurence(code_idx,
siblings_indices,co_occurrence_prob_matrix)
                    mx_scr = 0
                    mx_added_sib_idx = 0
                    for e, cooccured_sib_idx in enumerate(sibilings_cooccurence_indices):
                        if sibilings_cooccurence_probs[e] >= cooccrence_ratio:
                            if adv_x_test[visit_idx, cooccured_sib_idx] == 0:
                                adv_x_test_added_copy = adv_x_test.copy()
                                adv_x_test_added_copy[visit_idx, cooccured_sib_idx] = 1
                                test_added_scr = pred_duration(model, adv_x_test_added_copy).numpy()[0]
                                if replacing:
                                    if (test_added_scr < min(current_score, added_syn_score)) & (np.abs(test_added_scr - added_syn_score) > maximum):
                                        mx_scr = test_added_scr
                                        mx_added_sib_idx = cooccured_sib_idx

                                else:
                                    min_score = min(current_score, score)
                                    if (test_added_scr < min_score) & (np.abs(test_added_scr - min_score) > maximum):
                                        mx_scr = test_added_scr
                                        mx_added_sib_idx = cooccured_sib_idx

                    if mx_added_sib_idx != 0:
                        adv_x_test[visit_idx, mx_added_sib_idx] = 1
                        added_codes.append((visit_idx, code_idx, mx_added_sib_idx))
                        if mx_scr < comparing_score:
                            final_adv = adv_x_test
                            break


            # check
            final_iter_score = pred_duration(model, adv_x_test).numpy()[0]
            if final_iter_score < comparing_score:
                final_adv = adv_x_test
                break


        if type(final_adv) == int:
            print('attack failed')
            final_score = pred_duration(model, adv_x_test).numpy()[0]
            change_rate = 100 * (len(removed_codes) + len(added_codes)) / x_attacked_patient[:, 2:].sum()
        else:
            final_score = pred_duration(model, final_adv).numpy()[0]
            change_rate = 100 * (len(removed_codes) + len(added_codes)) / x_attacked_patient[:, 2:].sum()
            if verbose:
                print('Black-Box attack is done => patient pred-time orders has been attacked!')
                print(f'original pred-time orders: p1 :{comparing_score:.3f} < p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} > p2 :{final_score:.3f}')
                print(f'number of perturbations: {len(removed_codes) + len(added_codes)}')
                print(f'number of removed : {len(removed_codes) - len(replaced_codes)}')
                print(f'number of replaced: {len(replaced_codes)}')
                print(f'number of added: {len(added_codes)}')
                print(f'change rate: {change_rate:.2f}%')
                visit_of_change = np.array([item[0] for item in removed_codes] + [item[0] for item in added_codes])
                for i in range(num_visits):
                    print(f'number of changes in visit {i + 1} : {(visit_of_change == i).sum()} / {x_attacked_patient[i, 2:].sum()}')


    #increase
    else:
        for item in sorted_codes:
            removed = False
            visit_idx = item[0][0]
            code_idx = item[0][1]
            current_score = pred_duration(model, adv_x_test).numpy()[0]

            adv_x_test[visit_idx, code_idx] = 0
            score = pred_duration(model, adv_x_test).numpy()[0]

            if score > max(current_score, s_real):
                removed_codes.append((visit_idx, code_idx))
                removed = True

                if score > comparing_score:
                    final_adv = adv_x_test
                    break
            else:
                adv_x_test[visit_idx, code_idx] = 1

            # checking replacing

            if code_idx >= 1126:
                replacing = False
                # act3_class = onto_vocab_rx[onto_vocab_rx['atc4'] == rxdx_codes[code_idx]]['atc3'].values[0]
                # synonym_set = onto_vocab_rx[onto_vocab_rx['atc3']==act3_class]['atc4'].unique()
                synonym_set = get_synonym_codes(code_idx, True, onto_rx, rxdx_codes)

                ######################################### replacing
                chosen_syn = 0
                maximum = 0
                for synonym_code in synonym_set:
                    if synonym_code in rxdx_codes:
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx] = 0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]] = 1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp > current_score) & (np.abs(s_temp - current_score) > maximum):
                            maximum = np.abs(s_temp - current_score)
                            chosen_syn = synonym_code

                if chosen_syn != 0:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn]] == 0:
                        if adv_x_test[visit_idx, code_idx] == 0:
                            already_zero = 1
                        else:
                            already_zero = 0

                        adv_x_test[visit_idx, code_idx] = 0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]] = 1
                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score > max(s_real, current_score, score):
                            replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                            replacing = True
                            if already_zero == 0:
                                removed = True
                                removed_codes.append((visit_idx, code_idx))
                            if added_syn_score > comparing_score:
                                final_adv = adv_x_test
                                break
                        elif (added_syn_score > max(s_real, current_score)) & (added_syn_score < score):
                            adv_x_test[visit_idx, code_idx] = 0
                            removed = True
                            adv_x_test[visit_idx, code_ctoi[chosen_syn]] = 0
                        else:
                            if already_zero == 1:
                                adv_x_test[visit_idx, code_idx] = 0
                            else:
                                adv_x_test[visit_idx, code_idx] = 1

                            adv_x_test[visit_idx, code_ctoi[chosen_syn]] = 0

                # check
                # if pred_duration(model, adv_x_test).numpy()[0] < comparing_score:
                # final_adv = adv_x_test
                # break

                #########################################adding
                synonym_set_copy = synonym_set.copy()
                if replacing:
                    synonym_set_copy.remove(chosen_syn)

                siblings_indices = [code_ctoi[sibling] for sibling in synonym_set_copy if sibling in rxdx_codes]
                #includes first 2 indices

                if len(siblings_indices)!=0:
                    sibilings_cooccurence_indices, sibilings_cooccurence_probs = get_sort_cooccurence(code_idx,
                                                                 siblings_indices,co_occurrence_prob_matrix)
                    #sibilings_cooccurence_indices-->includes first 2 indices
                    mx_scr = 0
                    mx_added_sib_idx = 0
                    for e, cooccured_sib_idx in enumerate(sibilings_cooccurence_indices):
                        if sibilings_cooccurence_probs[e] >= cooccrence_ratio:
                            if adv_x_test[visit_idx, cooccured_sib_idx] == 0:
                                adv_x_test_added_copy = adv_x_test.copy()
                                adv_x_test_added_copy[visit_idx, cooccured_sib_idx] = 1
                                test_added_scr = pred_duration(model, adv_x_test_added_copy).numpy()[0]
                                if replacing:
                                    if (test_added_scr > max(current_score, added_syn_score)) & (np.abs(test_added_scr - added_syn_score) > mx_scr):
                                        mx_scr = test_added_scr
                                        mx_added_sib_idx = cooccured_sib_idx

                                else:
                                    min_score = max(current_score, score)
                                    if (test_added_scr > min_score) & (np.abs(test_added_scr - min_score) > mx_scr):
                                        mx_scr = test_added_scr
                                        mx_added_sib_idx = cooccured_sib_idx

                    if mx_added_sib_idx != 0:
                        adv_x_test[visit_idx, mx_added_sib_idx] = 1
                        added_codes.append((visit_idx, code_idx, mx_added_sib_idx))
                        if mx_scr > comparing_score:
                            final_adv = adv_x_test
                            break


            else:
                replacing = False
                # l2_class = onto_vocab_dx[onto_vocab_dx['l3'] == rxdx_codes[code_idx]]['l2'].values[0]
                # synonym_set_dx = onto_vocab_dx[onto_vocab_dx['l2'] == l2_class]['l3'].unique()
                synonym_set_dx = get_synonym_codes(code_idx, False, onto_dx, rxdx_codes)

                ######################################### replacing
                chosen_syn_dx = 0
                maximum = 0
                for synonym_code in synonym_set_dx:
                    if synonym_code in rxdx_codes:
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx] = 0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]] = 1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp > current_score) & (np.abs(s_temp - current_score) > maximum):
                            maximum = np.abs(s_temp - current_score)
                            chosen_syn_dx = synonym_code

                if chosen_syn_dx != 0:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]] == 0:
                        if adv_x_test[visit_idx, code_idx] == 0:
                            already_zero = 1
                        else:
                            already_zero = 0
                        adv_x_test[visit_idx, code_idx] = 0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]] = 1

                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score > max(s_real, current_score, score):
                            replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn_dx]))
                            replacing = True
                            if already_zero == 0:
                                removed_codes.append((visit_idx, code_idx))
                                removed = True

                            if added_syn_score > comparing_score:
                                final_adv = adv_x_test
                                break

                        elif (added_syn_score > max(s_real, current_score)) & (added_syn_score < score):
                            adv_x_test[visit_idx, code_idx] = 0
                            adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]] = 0
                        else:
                            if already_zero == 1:
                                adv_x_test[visit_idx, code_idx] = 0
                            else:
                                adv_x_test[visit_idx, code_idx] = 1
                            adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]] = 0

                ########################################  adding
                synonym_set_copy = synonym_set_dx.copy()
                if replacing:
                    synonym_set_copy.remove(chosen_syn_dx)
                siblings_indices = [code_ctoi[sibling] for sibling in synonym_set_copy if sibling in rxdx_codes]
                if len(siblings_indices)!=0:
                    sibilings_cooccurence_indices, sibilings_cooccurence_probs = get_sort_cooccurence(code_idx,
                                                                                                      siblings_indices,                                                                                   co_occurrence_prob_matrix)
                    mx_scr = 0
                    mx_added_sib_idx = 0
                    for e, cooccured_sib_idx in enumerate(sibilings_cooccurence_indices):
                        if sibilings_cooccurence_probs[e] >= cooccrence_ratio:
                            if adv_x_test[visit_idx, cooccured_sib_idx] == 0:
                                adv_x_test_added_copy = adv_x_test.copy()
                                adv_x_test_added_copy[visit_idx, cooccured_sib_idx] = 1
                                test_added_scr = pred_duration(model, adv_x_test_added_copy).numpy()[0]
                                if replacing:
                                    if (test_added_scr > max(current_score, added_syn_score)) & (np.abs(test_added_scr - added_syn_score) > mx_scr):
                                        mx_scr = test_added_scr
                                        mx_added_sib_idx = cooccured_sib_idx

                                else:
                                    max_score = max(current_score, score)
                                    if (test_added_scr > max_score) & (np.abs(test_added_scr - max_score) > mx_scr):
                                        mx_scr = test_added_scr
                                        mx_added_sib_idx = cooccured_sib_idx

                    if mx_added_sib_idx != 0:
                        adv_x_test[visit_idx, mx_added_sib_idx] = 1
                        added_codes.append((visit_idx, code_idx, mx_added_sib_idx))
                        if mx_scr > comparing_score:
                            final_adv = adv_x_test
                            break

            # check
            final_iter_score = pred_duration(model, adv_x_test).numpy()[0]
            if final_iter_score > comparing_score:
                final_adv = adv_x_test
                break

        if type(final_adv) == int:
            print('attack failed')
            final_score = pred_duration(model, adv_x_test).numpy()[0]
            change_rate = 100 * (len(removed_codes) + len(added_codes)) / x_attacked_patient[:, 2:].sum()
        else:
            final_score = pred_duration(model, final_adv).numpy()[0]
            change_rate = 100 * (len(removed_codes) + len(added_codes)) / x_attacked_patient[:, 2:].sum()
            if verbose:
                print('Black-Box attack is done => patient pred-time orders has been attacked!')
                print(
                    f'original pred-time orders: p1 :{comparing_score:.3f} > p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} < p2 :{final_score:.3f}')
                print(f'number of perturbations: {len(removed_codes) + len(added_codes)}')
                print(f'number of removed : {len(removed_codes) - len(replaced_codes)}')
                print(f'number of replaced: {len(replaced_codes)}')
                print(f'number of added: {len(added_codes)}')
                print(f'change rate: {change_rate:.2f}%')
                visit_of_change = np.array([item[0] for item in removed_codes] + [item[0] for item in added_codes])
                for i in range(num_visits):
                    print(f'number of changes in visit {i + 1} : {(visit_of_change == i).sum()} / {x_attacked_patient[i, 2:].sum()}')

    return final_adv, removed_codes, replaced_codes, final_score, change_rate, comparing_score, s_real

def Bbox_AdvAttack_SA_pair0(model, patient1, patient2, rxdx_codes, onto_rx, onto_dx, increase=False, verbose=True):
    """flip the predicted time/risk of two patients based on only code score"""

    if not increase:
        x_fixed_patient, x_attacked_patient = patient1, patient2
    else:
        x_attacked_patient, x_fixed_patient = patient1, patient2

    code_ctoi = {code:i for i, code in enumerate(rxdx_codes)}
    comparing_score = pred_duration(model, x_fixed_patient).numpy()[0]

    code_ds_dict, sorted_codes, s_real = patient_code_score2(model, x_attacked_patient, rxdx_codes[2:], increase=increase)
    num_visits = x_attacked_patient.shape[0]

    if (not increase) & (s_real < comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, comparing_score, s_real

    if (increase) & (s_real > comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, comparing_score, s_real

    adv_x_test = x_attacked_patient.copy()
    removed_codes = []
    replaced_codes = []
    final_adv = 0

    if not increase:
        for item in sorted_codes:
            visit_idx = item[0][0]
            code_idx = item[0][1]
            current_score = pred_duration(model, adv_x_test).numpy()[0]

            adv_x_test[visit_idx, code_idx]=0
            score = pred_duration(model, adv_x_test).numpy()[0]

            if score < min(current_score, s_real):
                removed_codes.append((visit_idx, code_idx))


                if score < comparing_score:
                    final_adv = adv_x_test
                    break

            else:
                adv_x_test[visit_idx, code_idx]=1

             #checking replacing

            if code_idx >= 1126:
                synonym_set = get_synonym_codes(code_idx, True, onto_rx, rxdx_codes)

                ######################################### replacing
                chosen_syn=0
                maximum = 0
                for synonym_code in synonym_set:
                    if synonym_code in rxdx_codes:
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx]=0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp < current_score) & (np.abs(s_temp-current_score) > maximum):
                            maximum = np.abs(s_temp-current_score)
                            chosen_syn = synonym_code

                if chosen_syn!=0:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn]]==0:
                        if adv_x_test[visit_idx, code_idx]==0:
                            already_zero=1
                        else:
                            already_zero=0

                        adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=1
                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score < min(s_real, current_score, score):
                            replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                            if already_zero==0:
                                removed_codes.append((visit_idx, code_idx))
                            if added_syn_score < comparing_score:
                                final_adv = adv_x_test
                                break
                        elif (added_syn_score < min(s_real, current_score)) & (added_syn_score > score):
                            adv_x_test[visit_idx, code_idx]=0
                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                        else:
                            if already_zero==1:
                                adv_x_test[visit_idx, code_idx]=0
                            else:
                                adv_x_test[visit_idx, code_idx]=1

                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0



            else:
                synonym_set_dx = get_synonym_codes(code_idx, False, onto_dx, rxdx_codes)

                #########################################replacing
                chosen_syn_dx=0
                maximum = 0
                for synonym_code in synonym_set_dx:
                    if synonym_code in rxdx_codes:
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx]=0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp < current_score) & (np.abs(s_temp-current_score) > maximum):
                            maximum = np.abs(s_temp-current_score)
                            chosen_syn_dx = synonym_code

                if chosen_syn_dx!=0:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]]==0:
                        if adv_x_test[visit_idx, code_idx]==0:
                            already_zero=1
                        else:
                            already_zero=0
                        adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]]=1
                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score < min(s_real, current_score, score):
                            replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn_dx]))
                            if already_zero==0:
                                removed_codes.append((visit_idx, code_idx))

                            if added_syn_score < comparing_score:
                                final_adv = adv_x_test
                                break

                        elif (added_syn_score < min(s_real, current_score)) & (added_syn_score > score):
                            adv_x_test[visit_idx, code_idx]=0
                            adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]]=0
                        else:
                            if already_zero==1:
                                adv_x_test[visit_idx, code_idx]=0
                            else:
                                adv_x_test[visit_idx, code_idx]=1
                            adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]]=0



            # check
            final_iter_score = pred_duration(model, adv_x_test).numpy()[0]
            if final_iter_score < comparing_score:
                final_adv = adv_x_test
                break


        if type(final_adv) == int:
            print('attack failed')
            final_score = pred_duration(model, adv_x_test).numpy()[0]
            change_rate = 100 * (len(removed_codes)) / x_attacked_patient[:, 2:].sum()
        else:
            final_score = pred_duration(model, final_adv).numpy()[0]
            change_rate = 100 * (len(removed_codes)) / x_attacked_patient[:, 2:].sum()
            if verbose:
                print('Black-Box attack is done => patient pred-time orders has been attacked!')
                print(f'original pred-time orders: p1 :{comparing_score:.3f} < p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} > p2 :{final_score:.3f}')
                print(f'number of perturbations: {len(removed_codes) }')
                print(f'number of removed : {len(removed_codes) - len(replaced_codes)}')
                print(f'number of replaced: {len(replaced_codes)}')
                print(f'change rate: {change_rate:.2f}%')
                visit_of_change = np.array([item[0] for item in removed_codes])
                for i in range(num_visits):
                    print(f'number of changes in visit {i + 1} : {(visit_of_change == i).sum()} / {x_attacked_patient[i, 2:].sum()}')


    #increase
    else:
        for item in sorted_codes:
            visit_idx = item[0][0]
            code_idx = item[0][1]
            current_score = pred_duration(model, adv_x_test).numpy()[0]

            adv_x_test[visit_idx, code_idx] = 0
            score = pred_duration(model, adv_x_test).numpy()[0]

            if score > max(current_score, s_real):
                removed_codes.append((visit_idx, code_idx))

                if score > comparing_score:
                    final_adv = adv_x_test
                    break
            else:
                adv_x_test[visit_idx, code_idx] = 1

            # checking replacing

            if code_idx >= 1126:
                synonym_set = get_synonym_codes(code_idx, True, onto_rx, rxdx_codes)

                ######################################### replacing
                chosen_syn = 0
                maximum = 0
                for synonym_code in synonym_set:
                    if synonym_code in rxdx_codes:
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx] = 0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]] = 1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp > current_score) & (np.abs(s_temp - current_score) > maximum):
                            maximum = np.abs(s_temp - current_score)
                            chosen_syn = synonym_code

                if chosen_syn != 0:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn]] == 0:
                        if adv_x_test[visit_idx, code_idx] == 0:
                            already_zero = 1
                        else:
                            already_zero = 0

                        adv_x_test[visit_idx, code_idx] = 0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]] = 1
                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score > max(s_real, current_score, score):
                            replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                            if already_zero == 0:
                                removed_codes.append((visit_idx, code_idx))
                            if added_syn_score > comparing_score:
                                final_adv = adv_x_test
                                break
                        elif (added_syn_score > max(s_real, current_score)) & (added_syn_score < score):
                            adv_x_test[visit_idx, code_idx] = 0
                            adv_x_test[visit_idx, code_ctoi[chosen_syn]] = 0
                        else:
                            if already_zero == 1:
                                adv_x_test[visit_idx, code_idx] = 0
                            else:
                                adv_x_test[visit_idx, code_idx] = 1

                            adv_x_test[visit_idx, code_ctoi[chosen_syn]] = 0




            else:
                synonym_set_dx = get_synonym_codes(code_idx, False, onto_dx, rxdx_codes)

                ######################################### replacing
                chosen_syn_dx = 0
                maximum = 0
                for synonym_code in synonym_set_dx:
                    if synonym_code in rxdx_codes:
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx] = 0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]] = 1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp > current_score) & (np.abs(s_temp - current_score) > maximum):
                            maximum = np.abs(s_temp - current_score)
                            chosen_syn_dx = synonym_code

                if chosen_syn_dx != 0:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]] == 0:
                        if adv_x_test[visit_idx, code_idx] == 0:
                            already_zero = 1
                        else:
                            already_zero = 0
                        adv_x_test[visit_idx, code_idx] = 0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]] = 1

                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score > max(s_real, current_score, score):
                            replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn_dx]))
                            if already_zero == 0:
                                removed_codes.append((visit_idx, code_idx))

                            if added_syn_score > comparing_score:
                                final_adv = adv_x_test
                                break

                        elif (added_syn_score > max(s_real, current_score)) & (added_syn_score < score):
                            adv_x_test[visit_idx, code_idx] = 0
                            adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]] = 0
                        else:
                            if already_zero == 1:
                                adv_x_test[visit_idx, code_idx] = 0
                            else:
                                adv_x_test[visit_idx, code_idx] = 1
                            adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]] = 0


        if type(final_adv) == int:
            print('attack failed')
            final_score = pred_duration(model, adv_x_test).numpy()[0]
            change_rate = 100 * (len(removed_codes)) / x_attacked_patient[:, 2:].sum()
        else:
            final_score = pred_duration(model, final_adv).numpy()[0]
            change_rate = 100 * (len(removed_codes)) / x_attacked_patient[:, 2:].sum()
            if verbose:
                print('Black-Box attack is done => patient pred-time orders has been attacked!')
                print(
                    f'original pred-time orders: p1 :{comparing_score:.3f} > p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} < p2 :{final_score:.3f}')
                print(f'number of perturbations: {len(removed_codes)}')
                print(f'number of removed : {len(removed_codes) - len(replaced_codes)}')
                print(f'number of replaced: {len(replaced_codes)}')
                print(f'change rate: {change_rate:.2f}%')
                visit_of_change = np.array([item[0] for item in removed_codes])
                for i in range(num_visits):
                    print(f'number of changes in visit {i + 1} : {(visit_of_change == i).sum()} / {x_attacked_patient[i, 2:].sum()}')

    return final_adv, removed_codes, replaced_codes, final_score, change_rate, comparing_score, s_real

############################################################################################################
def Bbox_AdvAttack_SA_pair_censor(model, patient1, patient2, rxdx_codes, onto_rx, onto_dx, co_occurrence_prob_matrix, cooccrence_ratio, increase=False, verbose=True):
    """attack function for a censored data; lowring the predicted time/risk by comparing it with an obsorved data, contingent on the change rate of less than x percent"""


    x_fixed_patient, x_attacked_patient = patient1, patient2
    code_ctoi = {code:i for i, code in enumerate(rxdx_codes)}
    comparing_score = pred_duration(model, x_fixed_patient).numpy()[0]

    code_ds_dict, sorted_codes, s_real = patient_code_score2(model, x_attacked_patient, rxdx_codes[2:], increase=increase)
    num_visits = x_attacked_patient.shape[0]

    if (not increase) & (s_real < comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, comparing_score, s_real


    adv_x_test = x_attacked_patient.copy()
    removed_codes = []
    replaced_codes = []
    added_codes = []
    final_adv = 0


    for item in sorted_codes:
        visit_idx = item[0][0]
        code_idx = item[0][1]

        current_score = pred_duration(model, adv_x_test).numpy()[0]
        adv_x_test[visit_idx, code_idx]=0
        score = pred_duration(model, adv_x_test).numpy()[0]
        if score < min(current_score, s_real):
            removed_codes.append((visit_idx, code_idx))

            change_rate = 100 * (len(removed_codes) + len(added_codes)) / x_attacked_patient[:, 2:].sum()
            if change_rate > 10:
                print('=============> break by change_rate limit')
                adv_x_test[visit_idx, code_idx]=1
                removed_codes.remove((visit_idx, code_idx))
                final_adv = adv_x_test
                break

            if score < comparing_score:
                final_adv = adv_x_test
                break

        else:
            adv_x_test[visit_idx, code_idx]=1

         #checking replacing
        if code_idx >= 1126:
            replacing = False
            synonym_set = get_synonym_codes(code_idx, True, onto_rx, rxdx_codes)

            chosen_syn=0
            maximum = 0
            for synonym_code in synonym_set:
                if synonym_code in rxdx_codes:
                    adv_x_test_copy = adv_x_test.copy()
                    adv_x_test_copy[visit_idx, code_idx]=0
                    adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                    s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                    if (s_temp < current_score) & (np.abs(s_temp-current_score) > maximum):
                        maximum = np.abs(s_temp-current_score)
                        chosen_syn = synonym_code

            if chosen_syn != 0:
                if adv_x_test[visit_idx, code_ctoi[chosen_syn]]==0:
                    if adv_x_test[visit_idx, code_idx]==0:
                        already_zero=1
                    else:
                        already_zero=0
                    adv_x_test[visit_idx, code_idx]=0
                    adv_x_test[visit_idx, code_ctoi[chosen_syn]]=1
                    added_syn_score = pred_duration(model, adv_x_test).numpy()[0]

                    if added_syn_score < min(s_real, current_score, score):
                        replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                        replacing = True
                        if already_zero==0:
                            removed_codes.append((visit_idx, code_idx))

                        change_rate = 100 * (len(removed_codes) + len(added_codes)) / x_attacked_patient[:, 2:].sum()
                        if change_rate > 10:
                            print('========================> break by change_rate limit')
                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                            replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn]))
                            if already_zero==0:
                                removed_codes.remove((visit_idx, code_idx))
                                adv_x_test[visit_idx, code_idx]=1
                            final_adv = adv_x_test
                            break

                        if added_syn_score < comparing_score:
                            final_adv = adv_x_test
                            break


                    elif (added_syn_score < min(s_real, current_score)) & (added_syn_score > score): #it means base code already removed
                        adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0

                    else:
                        if already_zero==1:
                            adv_x_test[visit_idx, code_idx]=0
                        else:
                            adv_x_test[visit_idx, code_idx]=1
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0

            ############adding
            synonym_set_copy=synonym_set.copy()
            if replacing:
                synonym_set_copy.remove(chosen_syn)
            siblings_indices = [code_ctoi[sibling] for sibling in synonym_set_copy if sibling in rxdx_codes]
            if len(siblings_indices)!=0:
                sibilings_cooccurence_indices, sibilings_cooccurence_probs = get_sort_cooccurence(code_idx, siblings_indices, co_occurrence_prob_matrix)
                mx_scr=0
                mx_added_sib_idx=0
                for e, cooccured_sib_idx in enumerate(sibilings_cooccurence_indices):
                    if sibilings_cooccurence_probs[e] >= cooccrence_ratio:
                        if adv_x_test[visit_idx, cooccured_sib_idx]==0:
                            adv_x_test_added_copy = adv_x_test.copy()
                            adv_x_test_added_copy[visit_idx, cooccured_sib_idx]=1
                            test_added_scr = pred_duration(model, adv_x_test_added_copy).numpy()[0]
                            if replacing:
                                if (test_added_scr < min(current_score, added_syn_score)) & (np.abs(test_added_scr - added_syn_score) > mx_scr):
                                    mx_scr = test_added_scr
                                    mx_added_sib_idx = cooccured_sib_idx

                            else:
                                min_score = min(current_score, score)
                                if (test_added_scr < min_score) & (np.abs(test_added_scr - min_score) > mx_scr):
                                    mx_scr = test_added_scr
                                    mx_added_sib_idx = cooccured_sib_idx

                if mx_added_sib_idx!=0:
                    adv_x_test[visit_idx, mx_added_sib_idx]=1
                    added_codes.append((visit_idx, code_idx, mx_added_sib_idx))
                    change_rate = 100 * (len(removed_codes) + len(added_codes)) / x_attacked_patient[:, 2:].sum()
                    if change_rate > 10:
                        print('========================> break by change_rate limit')
                        adv_x_test[visit_idx, mx_added_sib_idx] = 0
                        added_codes.remove((visit_idx, code_idx, mx_added_sib_idx))
                        final_adv = adv_x_test
                        break

                    if mx_scr < comparing_score:
                        final_adv = adv_x_test
                        break

        else:
            synonym_set_dx = get_synonym_codes(code_idx, False, onto_dx, rxdx_codes)

            chosen_syn_dx=0
            maximum = 0

            for synonym_code in synonym_set_dx:
                if synonym_code in rxdx_codes:
                    adv_x_test_copy = adv_x_test.copy()
                    adv_x_test_copy[visit_idx, code_idx] = 0
                    adv_x_test_copy[visit_idx, code_ctoi[synonym_code]] = 1
                    s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                    if (s_temp < current_score) & (np.abs(s_temp - current_score) > maximum):
                        maximum = np.abs(s_temp - current_score)
                        chosen_syn_dx = synonym_code

            if chosen_syn_dx != 0:
                if adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]] == 0:
                    if adv_x_test[visit_idx, code_idx] == 0:
                        already_zero = 1
                    else:
                        already_zero = 0
                    adv_x_test[visit_idx, code_idx] = 0
                    adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]] = 1
                    added_syn_score = pred_duration(model, adv_x_test).numpy()[0]

                    if added_syn_score < min(s_real, current_score, score):
                        replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn_dx]))
                        replacing = True
                        if already_zero == 0:
                            removed_codes.append((visit_idx, code_idx))

                        change_rate = 100 * (len(removed_codes) + len(added_codes)) / x_attacked_patient[:, 2:].sum()
                        if change_rate > 10:
                            print('========================> break by change_rate limit')
                            adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]] = 0
                            replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn_dx]))
                            if already_zero == 0:
                                removed_codes.remove((visit_idx, code_idx))
                                adv_x_test[visit_idx, code_idx] = 1
                            final_adv = adv_x_test
                            break

                        if added_syn_score < comparing_score:
                            final_adv = adv_x_test
                            break


                    elif (added_syn_score < min(s_real, current_score)) & (
                            added_syn_score > score):  # it means base code already removed
                        adv_x_test[visit_idx, code_idx] = 0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]] = 0

                    else:
                        if already_zero == 1:
                            adv_x_test[visit_idx, code_idx] = 0
                        else:
                            adv_x_test[visit_idx, code_idx] = 1
                        adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]] = 0

            synonym_set_copy = synonym_set_dx.copy()
            if replacing:
                synonym_set_copy.remove(chosen_syn_dx)
            siblings_indices = [code_ctoi[sibling] for sibling in synonym_set_copy if sibling in rxdx_codes]
            if len(siblings_indices) != 0:
                sibilings_cooccurence_indices, sibilings_cooccurence_probs = get_sort_cooccurence(code_idx, siblings_indices,co_occurrence_prob_matrix)
                mx_scr = 0
                mx_added_sib_idx = 0
                for e, cooccured_sib_idx in enumerate(sibilings_cooccurence_indices):
                    if sibilings_cooccurence_probs[e] >= 0.10:
                        if adv_x_test[visit_idx, cooccured_sib_idx] == 0:
                            adv_x_test_added_copy = adv_x_test.copy()
                            adv_x_test_added_copy[visit_idx, cooccured_sib_idx] = 1
                            test_added_scr = pred_duration(model, adv_x_test_added_copy).numpy()[0]
                            if replacing:
                                if (test_added_scr < min(current_score, added_syn_score)) & (
                                        np.abs(test_added_scr - added_syn_score) > mx_scr):
                                    mx_scr = test_added_scr
                                    mx_added_sib_idx = cooccured_sib_idx

                            else:
                                min_score = min(current_score, score)
                                if (test_added_scr < min_score) & (np.abs(test_added_scr - min_score) > mx_scr):
                                    mx_scr = test_added_scr
                                    mx_added_sib_idx = cooccured_sib_idx

                if mx_added_sib_idx != 0:
                    adv_x_test[visit_idx, mx_added_sib_idx] = 1
                    added_codes.append((visit_idx, code_idx, mx_added_sib_idx))

                    change_rate = 100 * (len(removed_codes) + len(added_codes)) / x_attacked_patient[:, 2:].sum()
                    if change_rate > 10:
                        print('========================> break by change_rate limit')
                        adv_x_test[visit_idx, mx_added_sib_idx] = 0
                        added_codes.remove((visit_idx, code_idx, mx_added_sib_idx))
                        final_adv = adv_x_test
                        break

                    if mx_scr < comparing_score:
                        final_adv = adv_x_test
                        break
        # check
        final_iter_score = pred_duration(model, adv_x_test).numpy()[0]
        if final_iter_score < comparing_score:
            final_adv = adv_x_test
            break


    if type(final_adv)==int:
        print('attack failed')
        final_score = pred_duration(model, adv_x_test).numpy()[0]
        change_rate = 100 * (len(removed_codes) + len(added_codes)) / x_attacked_patient[:, 2:].sum()
    else:
        final_score = pred_duration(model, final_adv).numpy()[0]
        change_rate = 100 * (len(removed_codes) + len(added_codes)) / x_attacked_patient[:, 2:].sum()
        if verbose:
            print('Black-Box attack is done => patient pred-time orders has been attacked!')
            print(f'original pred-time orders: p1 :{comparing_score:.3f} < p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} > p2 :{final_score:.3f}')
            print(f'number of perturbations: {len(removed_codes) + len(added_codes)}')
            print(f'number of removed: {len(removed_codes) - len(replaced_codes)}')
            print(f'number of replaced: {len(replaced_codes)}')
            print(f'number of added: {len(added_codes)}')
            print(f'change rate: {change_rate:.2f}%')
            visit_of_change= np.array([item[0] for item in removed_codes] + [item[0] for item in added_codes])
            for i in range(num_visits):
                print(f'number of changes in visit {i+1} : {(visit_of_change==i).sum()} / {x_attacked_patient[i, 2:].sum()}')


    return final_adv, removed_codes, replaced_codes, final_score, change_rate, comparing_score, s_real

def Bbox_AdvAttack_SA_pair_censor0(model, x_attacked_patient, rxdx_codes, onto_rx, onto_dx, ch_rate=15, increase=False, verbose=True):
    """attack function for a censored data; lowring the predicted time/risk by comparing it with an obsorved data, contingent on the change rate of less than x percent"""

    code_ctoi = {code:i for i, code in enumerate(rxdx_codes)}
    comparing_score = 0

    code_ds_dict, sorted_codes, s_real = patient_code_score2(model, x_attacked_patient, rxdx_codes[2:], increase=increase)
    num_visits = x_attacked_patient.shape[0]

    if (not increase) & (s_real < comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, comparing_score, s_real


    adv_x_test = x_attacked_patient.copy()
    removed_codes = []
    replaced_codes = []
    final_adv = 0


    for item in sorted_codes:
        visit_idx = item[0][0]
        code_idx = item[0][1]

        current_score = pred_duration(model, adv_x_test).numpy()[0]
        adv_x_test[visit_idx, code_idx]=0
        score = pred_duration(model, adv_x_test).numpy()[0]

        if score < min(current_score, s_real):
            removed_codes.append((visit_idx, code_idx))

            change_rate = 100 * (len(removed_codes) - 0.5*len(replaced_codes)) / x_attacked_patient[:, 2:].sum()
            if change_rate > ch_rate:
                print('=============> break by change_rate limit on remove')
                adv_x_test[visit_idx, code_idx]=1
                removed_codes.remove((visit_idx, code_idx))
                final_adv = adv_x_test
                break

            if score < comparing_score:
                final_adv = adv_x_test
                break

        else:
            adv_x_test[visit_idx, code_idx]=1

         # checking replacing
        if code_idx >= 1126:
            synonym_set = get_synonym_codes(code_idx, True, onto_rx, rxdx_codes)
            chosen_syn=0
            maximum = 0
            for synonym_code in synonym_set:
                if synonym_code in rxdx_codes:
                    adv_x_test_copy = adv_x_test.copy()
                    adv_x_test_copy[visit_idx, code_idx]=0
                    adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                    s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                    if (s_temp < current_score) & (np.abs(s_temp-current_score) > maximum):
                        maximum = np.abs(s_temp-current_score)
                        chosen_syn = synonym_code

            if chosen_syn != 0:
                if adv_x_test[visit_idx, code_ctoi[chosen_syn]]==0:
                    if adv_x_test[visit_idx, code_idx]==0:
                        already_zero=1
                    else:
                        already_zero=0
                    adv_x_test[visit_idx, code_idx]=0
                    adv_x_test[visit_idx, code_ctoi[chosen_syn]]=1
                    added_syn_score = pred_duration(model, adv_x_test).numpy()[0]

                    if added_syn_score < min(s_real, current_score, score):
                        replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                        if already_zero==0:
                            removed_codes.append((visit_idx, code_idx))

                        change_rate = 100 * (len(removed_codes) - 0.5*len(replaced_codes)) / x_attacked_patient[:, 2:].sum()
                        if change_rate > ch_rate:
                            print('========================> break by change_rate limit on replace rx')
                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                            replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn]))
                            if already_zero==0:
                                removed_codes.remove((visit_idx, code_idx))
                                adv_x_test[visit_idx, code_idx]=1
                            final_adv = adv_x_test
                            break

                        if added_syn_score < comparing_score:
                            final_adv = adv_x_test
                            break


                    elif (added_syn_score < min(s_real, current_score)) & (added_syn_score > score): #it means base code already removed
                        #adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0

                    else:
                        if already_zero==1:
                            adv_x_test[visit_idx, code_idx]=0
                        else:
                            adv_x_test[visit_idx, code_idx]=1
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0


        else:
            synonym_set_dx = get_synonym_codes(code_idx, False, onto_dx, rxdx_codes)

            chosen_syn_dx=0
            maximum = 0

            for synonym_code in synonym_set_dx:
                if synonym_code in rxdx_codes:
                    adv_x_test_copy = adv_x_test.copy()
                    adv_x_test_copy[visit_idx, code_idx] = 0
                    adv_x_test_copy[visit_idx, code_ctoi[synonym_code]] = 1
                    s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                    if (s_temp < current_score) & (np.abs(s_temp - current_score) > maximum):
                        maximum = np.abs(s_temp - current_score)
                        chosen_syn_dx = synonym_code

            if chosen_syn_dx != 0:
                if adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]] == 0:
                    if adv_x_test[visit_idx, code_idx] == 0:
                        already_zero = 1
                    else:
                        already_zero = 0
                    adv_x_test[visit_idx, code_idx] = 0
                    adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]] = 1
                    added_syn_score = pred_duration(model, adv_x_test).numpy()[0]

                    if added_syn_score < min(s_real, current_score, score):
                        replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn_dx]))
                        if already_zero == 0:
                            removed_codes.append((visit_idx, code_idx))

                        change_rate = 100 * (len(removed_codes) - 0.5*len(replaced_codes)) / x_attacked_patient[:, 2:].sum()
                        if change_rate > ch_rate:
                            print('========================> break by change_rate limit on replace')
                            adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]] = 0
                            replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn_dx]))
                            if already_zero == 0:
                                removed_codes.remove((visit_idx, code_idx))
                                adv_x_test[visit_idx, code_idx] = 1
                            final_adv = adv_x_test
                            break

                        if added_syn_score < comparing_score:
                            final_adv = adv_x_test
                            break


                    elif (added_syn_score < min(s_real, current_score)) & (
                            added_syn_score > score):  # it means base code already removed
                        adv_x_test[visit_idx, code_idx] = 0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]] = 0

                    else:
                        if already_zero == 1:
                            adv_x_test[visit_idx, code_idx] = 0
                        else:
                            adv_x_test[visit_idx, code_idx] = 1
                        adv_x_test[visit_idx, code_ctoi[chosen_syn_dx]] = 0



    if type(final_adv)==int:
        print('attack failed')
        final_score = pred_duration(model, adv_x_test).numpy()[0]
        change_rate = 100 * (len(removed_codes) - 0.5*len(replaced_codes)) / x_attacked_patient[:, 2:].sum()
    else:
        final_score = pred_duration(model, final_adv).numpy()[0]
        change_rate = 100 * (len(removed_codes) - 0.5*len(replaced_codes)) / x_attacked_patient[:, 2:].sum()
        if verbose:
            print('Black-Box attack is done => patient pred-time orders has been attacked!')
            print(f'original pred-time orders: p1 :{comparing_score:.3f} < p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} > p2 :{final_score:.3f}')
            print(f'number of perturbations: {len(removed_codes)}')
            print(f'number of removed: {len(removed_codes) - len(replaced_codes)}')
            print(f'number of replaced: {len(replaced_codes)}')
            print(f'change rate: {change_rate:.2f}%')
            visit_of_change= np.array([item[0] for item in removed_codes])
            for i in range(num_visits):
                print(f'number of changes in visit {i+1} : {(visit_of_change==i).sum()} / {x_attacked_patient[i, 2:].sum()}')


    return final_adv, removed_codes, replaced_codes, final_score, change_rate, comparing_score, s_real

def Bbox_AdvAttack_SA_pair_censored_new(model, patient1, patient2, rxdx_codes, onto_rx, onto_dx, co_occurrence_prob_matrix, cooccrence_ratio=0.15, ch_rate=15, increase=False, verbose=True):

    """flip the predicted time/risk of two patients based on only code score
    removing + replacing + adding
    search space is based on the scores of both existing codes and potential adding codes
    """

    if not increase:
        x_fixed_patient, x_attacked_patient = patient1, patient2
    else:
        x_attacked_patient, x_fixed_patient = patient1, patient2

    #code_itoc = {i:code for i, code in enumerate(rxdx_codes)}
    code_ctoi = {code:i for i, code in enumerate(rxdx_codes)}  #includes first 2 indices
    comparing_score = pred_duration(model, x_fixed_patient).numpy()[0]


    #the new sorting consider both
    code_ds_dict, sorted_codes, s_real = patient_code_score_new(model,
                                                                x_attacked_patient,
                                                            rxdx_codes[2:],
                                                                onto_rx,
                                                                onto_dx,
                                                                code_ctoi,
                                                                co_occurrence_prob_matrix,
                                                                p_cooccur=cooccrence_ratio,
                                                                increase=increase)
    num_visits = x_attacked_patient.shape[0]

    if (not increase) & (s_real < comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, None, comparing_score, s_real

    if (increase) & (s_real > comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, None, comparing_score, s_real

    adv_x_test = x_attacked_patient.copy()
    removed_codes = []
    replaced_codes = []
    added_codes = []
    final_adv = 0

    if not increase:
        for item in sorted_codes:

            visit_idx = item[0][0]
            label = item[0][1]
            code_idx = item[0][2]

            current_score = pred_duration(model, adv_x_test).numpy()[0]

            if label == 'remove':
                if adv_x_test[visit_idx, code_idx]==1:
                    adv_x_test[visit_idx, code_idx]=0
                    score_removed = pred_duration(model, adv_x_test).numpy()[0]

                    if score_removed < min(current_score, s_real):
                        removed_codes.append((visit_idx, code_idx))
                        change_rate = 100 * (len(removed_codes) + len(added_codes)) / x_attacked_patient[:, 2:].sum()
                        if change_rate > ch_rate:
                            print('=============> break by change_rate limit')
                            adv_x_test[visit_idx, code_idx]=1
                            removed_codes.remove((visit_idx, code_idx))
                            final_adv = adv_x_test
                            break
                        #check attack
                        if score_removed < comparing_score:
                            final_adv = adv_x_test
                            break


                    else:
                        adv_x_test[visit_idx, code_idx]=1

                 # checking replacing
                #if code_idx >= 1126:
                synonym_set = item[0][3]
                #synonym_set = get_synonym_codes(code_idx, True, onto_vocab_rx, rxdx_codes)
                ##################### replacing
                chosen_syn=0
                maximum=0
                for synonym_code in synonym_set:
                    if synonym_code in rxdx_codes:
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx]=0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp < current_score) & (np.abs(s_temp-current_score) > maximum):
                            maximum = np.abs(s_temp-current_score)
                            chosen_syn = synonym_code

                if chosen_syn!=0:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn]]==0:
                        if adv_x_test[visit_idx, code_idx]==0:
                            already_zero=1
                        else:
                            already_zero=0

                        adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=1
                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score < min(s_real, current_score, score_removed):
                            replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                            if already_zero==0:
                                removed_codes.append((visit_idx, code_idx))

                            change_rate = 100 * (len(removed_codes) + len(added_codes)) / x_attacked_patient[:, 2:].sum()
                            if change_rate > ch_rate:
                                print('========================> break by change_rate limit')
                                adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                                replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn]))
                                if already_zero==0:
                                    removed_codes.remove((visit_idx, code_idx))
                                    adv_x_test[visit_idx, code_idx]=1
                                final_adv = adv_x_test
                                break


                            if added_syn_score < comparing_score:
                                final_adv = adv_x_test
                                break
                        elif (added_syn_score < min(s_real, current_score)) & (added_syn_score > score_removed):
                            adv_x_test[visit_idx, code_idx]=0
                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                        else:
                            if already_zero==1:
                                adv_x_test[visit_idx, code_idx]=0
                            else:
                                adv_x_test[visit_idx, code_idx]=1

                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0

            else: #label=='add'
                origin = item[0][3]
                if adv_x_test[visit_idx, code_idx]==0:
                    adv_x_test[visit_idx, code_idx]=1
                    score_added = pred_duration(model, adv_x_test).numpy()[0]

                    if score_added < min(current_score, s_real):
                        added_codes.append((visit_idx, origin, code_idx))

                        change_rate = 100 * (len(removed_codes) + len(added_codes)) / x_attacked_patient[:, 2:].sum()
                        if change_rate > ch_rate:
                            print('=============> break by change_rate limit')
                            adv_x_test[visit_idx, code_idx]=0
                            added_codes.remove((visit_idx, origin, code_idx))
                            final_adv = adv_x_test
                            break

                        #check attack
                        if score_added < comparing_score:
                            final_adv = adv_x_test
                            break
                    else:
                        adv_x_test[visit_idx, code_idx]=0

        if type(final_adv) == int:
            print('attack failed')
            final_score = pred_duration(model, adv_x_test).numpy()[0]
            change_rate = 100 * (len(added_codes) + len(removed_codes)) / x_attacked_patient[:, 2:].sum()
        else:
            final_score = pred_duration(model, final_adv).numpy()[0]
            change_rate = 100 * (len(added_codes) + len(removed_codes)) / x_attacked_patient[:, 2:].sum()
            if verbose:
                print('Black-Box attack is done => patient pred-time orders has been attacked!')
                print(f'original pred-time orders: p1 :{comparing_score:.3f} < p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} > p2 :{final_score:.3f}')
                print(f'number of perturbations: {len(removed_codes) + len(added_codes)}')
                print(f'number of removed : {len(removed_codes) - len(replaced_codes)}')
                print(f'number of replaced: {len(replaced_codes)}')
                print(f'number of added: {len(added_codes)}')
                print(f'change rate: {change_rate:.2f}%')
                visit_of_change = np.array([item[0] for item in removed_codes] + [item[0] for item in added_codes])
                for i in range(num_visits):
                    print(f'number of changes in visit {i + 1} : {(visit_of_change == i).sum()} / {x_attacked_patient[i, 2:].sum()}')

    return final_adv, removed_codes, replaced_codes, added_codes, final_score, change_rate, comparing_score, s_real

############################ new-versions
def Bbox_AdvAttack_SA_pai0_new(model, x_attacked_patient, comparing_time, rxdx_codes, onto_rx, onto_dx, similarity_limit=0.85, increase=False, verbose=True):
    """attack function for a observed data point, first consider remove and then replace"""


    code_ctoi = {code:i for i, code in enumerate(rxdx_codes)}
    comparing_score = comparing_time

    code_ds_dict, sorted_codes, s_real = patient_code_score2(model, x_attacked_patient, rxdx_codes[2:], increase=increase)
    num_visits = x_attacked_patient.shape[0]

    if (not increase) & (s_real < comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, None, None

    if (increase) & (s_real > comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, None, None


    adv_x_test = x_attacked_patient.copy()
    removed_codes = []
    replaced_codes = []
    final_adv = 0
    flag = 'n'


    if not increase:
        for item in sorted_codes:
            visit_idx = item[0][0]
            code_idx = item[0][1]

            current_score = pred_duration(model, adv_x_test).numpy()[0]
            adv_x_test[visit_idx, code_idx]=0
            score = pred_duration(model, adv_x_test).numpy()[0]

            if score < min(current_score, s_real):
                removed_codes.append((visit_idx, code_idx))

                #change_rate = 100 * (len(removed_codes) - 0.5*len(replaced_codes)) / x_attacked_patient[:, 2:].sum()
                similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                #if change_rate > ch_rate:
                if similarity < similarity_limit:
                    print('=============> passing similarity_limit on removal')
                    adv_x_test[visit_idx, code_idx]=1
                    removed_codes.remove((visit_idx, code_idx))
                    #final_adv = adv_x_test
                    score = current_score
                    #flag = 0
                    #break
                else:
                    if score < comparing_score:
                        print('=============> break by time limit on removal')
                        adv_x_test[visit_idx, code_idx]=1
                        removed_codes.remove((visit_idx, code_idx))
                        final_adv = adv_x_test
                        flag=1
                        break

            else:
                adv_x_test[visit_idx, code_idx]=1

             # checking replacing
            if code_idx >= 1126:
                synonym_set = get_synonym_codes(code_idx, True, onto_rx, rxdx_codes)
            else:
                synonym_set = get_synonym_codes(code_idx, False, onto_dx, rxdx_codes)


            chosen_syn=0
            maximum = 0
            for synonym_code in synonym_set:
                if synonym_code in rxdx_codes:
                    adv_x_test_copy = adv_x_test.copy()
                    adv_x_test_copy[visit_idx, code_idx]=0
                    adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                    s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                    if (s_temp < current_score) & (np.abs(s_temp-current_score) > maximum):
                        maximum = np.abs(s_temp-current_score)
                        chosen_syn = synonym_code

            if chosen_syn != 0:
                if adv_x_test[visit_idx, code_ctoi[chosen_syn]]==0:
                    if adv_x_test[visit_idx, code_idx]==0:
                        already_zero=1
                    else:
                        already_zero=0
                    adv_x_test[visit_idx, code_idx]=0
                    adv_x_test[visit_idx, code_ctoi[chosen_syn]]=1
                    added_syn_score = pred_duration(model, adv_x_test).numpy()[0]

                    if added_syn_score < min(s_real, current_score, score):
                        replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                        if already_zero==0:
                            removed_codes.append((visit_idx, code_idx))

                        #change_rate = 100 * (len(removed_codes) - 0.5*len(replaced_codes)) / x_attacked_patient[:, 2:].sum()
                        similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                        #if change_rate > ch_rate:
                        if similarity < similarity_limit:
                            print('========================> break by similarity_limit on replace')
                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                            replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn]))
                            if already_zero==0:
                                removed_codes.remove((visit_idx, code_idx))
                                adv_x_test[visit_idx, code_idx]=1
                            final_adv = adv_x_test
                            flag=0
                            break
                        else:
                            if added_syn_score < comparing_score:
                                print('========================> break by time limit on replace')
                                adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                                replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn]))
                                if already_zero==0:
                                    removed_codes.remove((visit_idx, code_idx))
                                    adv_x_test[visit_idx, code_idx]=1
                                final_adv = adv_x_test
                                flag=1
                                break


                    elif (added_syn_score < min(s_real, current_score)) & (added_syn_score > score): #it means base code already removed
                        #adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0

                    else:
                        if already_zero==1:
                            adv_x_test[visit_idx, code_idx]=0
                        else:
                            adv_x_test[visit_idx, code_idx]=1
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0

    else:
        for item in sorted_codes:
            visit_idx = item[0][0]
            code_idx = item[0][1]

            current_score = pred_duration(model, adv_x_test).numpy()[0]
            adv_x_test[visit_idx, code_idx]=0
            score = pred_duration(model, adv_x_test).numpy()[0]

            if score > max(current_score, s_real):
                removed_codes.append((visit_idx, code_idx))

                #change_rate = 100 * (len(removed_codes) - 0.5 * len(replaced_codes)) / x_attacked_patient[:, 2:].sum()
                similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                if similarity < similarity_limit:
                    print('=============> passing similarity_limit on removal')
                    adv_x_test[visit_idx, code_idx]=1
                    removed_codes.remove((visit_idx, code_idx))
                    #final_adv = adv_x_test
                    score = current_score
                    #flag=0
                    #break

                else:
                    if score > comparing_score:
                        print('=============> break by time limit on removal')
                        final_adv = adv_x_test
                        flag=1
                        break

            else:
                adv_x_test[visit_idx, code_idx]=1

             # checking replacing
            if code_idx >= 1126:
                synonym_set = get_synonym_codes(code_idx, True, onto_rx, rxdx_codes)
            else:
                synonym_set = get_synonym_codes(code_idx, False, onto_dx, rxdx_codes)


            chosen_syn=0
            maximum = 0
            for synonym_code in synonym_set:
                if synonym_code in rxdx_codes:
                    adv_x_test_copy = adv_x_test.copy()
                    adv_x_test_copy[visit_idx, code_idx]=0
                    adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                    s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                    if (s_temp > current_score) & (np.abs(s_temp-current_score) > maximum):
                        maximum = np.abs(s_temp-current_score)
                        chosen_syn = synonym_code

            if chosen_syn != 0:
                if adv_x_test[visit_idx, code_ctoi[chosen_syn]]==0:
                    if adv_x_test[visit_idx, code_idx]==0:
                        already_zero=1
                    else:
                        already_zero=0
                    adv_x_test[visit_idx, code_idx]=0
                    adv_x_test[visit_idx, code_ctoi[chosen_syn]]=1
                    added_syn_score = pred_duration(model, adv_x_test).numpy()[0]

                    if added_syn_score > max(s_real, current_score, score):
                        replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                        if already_zero==0:
                            removed_codes.append((visit_idx, code_idx))

                        #change_rate = 100 * (len(removed_codes) - 0.5*len(replaced_codes)) / x_attacked_patient[:, 2:].sum()
                        similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                        if similarity < similarity_limit:
                            print('========================> break by similarity-limit on replace')
                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                            replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn]))
                            if already_zero==0:
                                removed_codes.remove((visit_idx, code_idx))
                                adv_x_test[visit_idx, code_idx]=1
                            final_adv = adv_x_test
                            flag=0
                            break

                        else:
                            if added_syn_score > comparing_score:
                                print('==================> break by passing the time limit on replace')
                                final_adv = adv_x_test
                                flag=1
                                break


                    elif (added_syn_score > max(s_real, current_score)) & (added_syn_score < score): #it means base code already removed
                        #adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0

                    else:
                        if already_zero==1:
                            adv_x_test[visit_idx, code_idx]=0
                        else:
                            adv_x_test[visit_idx, code_idx]=1
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0



    final_score = pred_duration(model, adv_x_test).numpy()[0]
    final_adv = adv_x_test

    if not increase:
        if final_score >= s_real:
            print('attack failed')
            similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
        else:
            final_score = pred_duration(model, final_adv).numpy()[0]
            similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
            #change_rate = 100 * (len(removed_codes) - 0.5*len(replaced_codes)) / x_attacked_patient[:, 2:].sum()
            if verbose:
                print('Black-Box attack is done => patient pred-time orders has been attacked!')
                print(f'original pred-time orders: p1 :{comparing_score:.3f} -- p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} -- p2 :{final_score:.3f}')
                print(f'number of perturbations: {len(removed_codes)}')
                print(f'number of removed: {len(removed_codes) - len(replaced_codes)}')
                print(f'number of replaced: {len(replaced_codes)}')
                print(f'similarity rate: {similarity:.4f}%')
                visit_of_change= np.array([item[0] for item in removed_codes])
                for i in range(num_visits):
                    print(f'number of changes in visit {i+1} : {(visit_of_change==i).sum()} / {x_attacked_patient[i, 2:].sum()}')

    else:
        if final_score <= s_real:
            print('attack failed')
            similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)

        else:
            final_score = pred_duration(model, final_adv).numpy()[0]
            similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
            #change_rate = 100 * (len(removed_codes) - 0.5*len(replaced_codes)) / x_attacked_patient[:, 2:].sum()
            if verbose:
                print('Black-Box attack is done => patient pred-time orders has been attacked!')
                print(f'original pred-time orders: p1 :{comparing_score:.3f} -- p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} -- p2 :{final_score:.3f}')
                print(f'number of perturbations: {len(removed_codes)}')
                print(f'number of removed: {len(removed_codes) - len(replaced_codes)}')
                print(f'number of replaced: {len(replaced_codes)}')
                print(f'similarity rate: {similarity:.4f}%')
                visit_of_change= np.array([item[0] for item in removed_codes])
                for i in range(num_visits):
                    print(f'number of changes in visit {i+1} : {(visit_of_change==i).sum()} / {x_attacked_patient[i, 2:].sum()}')

    return final_adv, removed_codes, replaced_codes, final_score, similarity, s_real, flag

def Bbox_AdvAttack_SA_pair_new_new(model, x_attacked_patient, comparing_time, rxdx_codes, onto_rx, onto_dx, co_occurrence_prob_matrix, cooccrence_ratio=0.15, similarity_limit=0.85, increase=False, verbose=True):

    """flip the predicted time/risk of two patients based on only code score
    removing + replacing + adding
    search space is based on the scores of both existing codes and potential adding codes
    """
    flag = 'n'
    code_ctoi = {code:i for i, code in enumerate(rxdx_codes)}
    comparing_score = comparing_time

    # the new sorting consider both
    code_ds_dict, sorted_codes, s_real = patient_code_score_new(model,
                                                            x_attacked_patient,
                                                            rxdx_codes[2:],
                                                            onto_rx,
                                                            onto_dx,
                                                            code_ctoi,
                                                            co_occurrence_prob_matrix,
                                                            p_cooccur=cooccrence_ratio,
                                                            increase=increase)
    num_visits = x_attacked_patient.shape[0]

    if (not increase) & (s_real < comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, None, None, None

    if (increase) & (s_real > comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, None, None, None

    adv_x_test = x_attacked_patient.copy()
    removed_codes = []
    replaced_codes = []
    added_codes = []
    final_adv = 0

    if not increase:
        for item in sorted_codes:

            visit_idx = item[0][0]
            label = item[0][1]
            code_idx = item[0][2]

            current_score = pred_duration(model, adv_x_test).numpy()[0]

            if label == 'remove':
                if adv_x_test[visit_idx, code_idx]==1:
                    adv_x_test[visit_idx, code_idx]=0
                    score_removed = pred_duration(model, adv_x_test).numpy()[0]

                    if score_removed < min(current_score, s_real):
                        removed_codes.append((visit_idx, code_idx))

                        #check similarity
                        similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                        #if change_rate > ch_rate:
                        if similarity < similarity_limit:
                            print('=============> passing similarity limit on removal')
                            adv_x_test[visit_idx, code_idx]=1
                            removed_codes.remove((visit_idx, code_idx))
                            score = current_score
                            #final_adv = adv_x_test
                            #flag = 0
                            #break

                        #check attack
                        else:
                            if score_removed < comparing_score:
                                print('=============> break by time limit on removal')
                                adv_x_test[visit_idx, code_idx]=1
                                removed_codes.remove((visit_idx, code_idx))
                                final_adv = adv_x_test
                                flag=1
                                break

                    else:
                        adv_x_test[visit_idx, code_idx]=1


                 # checking replacing
                #if code_idx >= 1126:
                synonym_set = item[0][3]
                #synonym_set = get_synonym_codes(code_idx, True, onto_vocab_rx, rxdx_codes)
                ##################### replacing
                chosen_syn=0
                maximum=0
                for synonym_code in synonym_set:
                    if synonym_code in rxdx_codes:
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx]=0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp < current_score) & (np.abs(s_temp-current_score) > maximum):
                            maximum = np.abs(s_temp-current_score)
                            chosen_syn = synonym_code

                if chosen_syn!=0:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn]]==0:
                        if adv_x_test[visit_idx, code_idx]==0:
                            already_zero=1
                        else:
                            already_zero=0

                        adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=1
                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score < min(s_real, current_score, score_removed):
                            replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                            if already_zero==0:
                                removed_codes.append((visit_idx, code_idx))

                            # check similarity
                            similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                            #if change_rate > ch_rate:
                            if similarity < similarity_limit:
                                print('========================> break by similarity-limit on replace')
                                adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                                replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn]))
                                if already_zero==0:
                                    removed_codes.remove((visit_idx, code_idx))
                                    adv_x_test[visit_idx, code_idx]=1
                                final_adv = adv_x_test
                                flag=0
                                break

                            else:
                                if added_syn_score < comparing_score:
                                    print('========================> break by time limit on replace')
                                    adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                                    replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn]))
                                    if already_zero==0:
                                        removed_codes.remove((visit_idx, code_idx))
                                        adv_x_test[visit_idx, code_idx]=1
                                    final_adv = adv_x_test
                                    flag=1
                                    break


                        elif (added_syn_score < min(s_real, current_score)) & (added_syn_score > score_removed):
                            adv_x_test[visit_idx, code_idx]=0
                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                        else:
                            if already_zero==1:
                                adv_x_test[visit_idx, code_idx]=0
                            else:
                                adv_x_test[visit_idx, code_idx]=1

                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0

            else: #label=='add'
                origin = item[0][3]
                if adv_x_test[visit_idx, code_idx]==0:
                    adv_x_test[visit_idx, code_idx]=1
                    score_added = pred_duration(model, adv_x_test).numpy()[0]

                    if score_added < min(current_score, s_real):
                        added_codes.append((visit_idx, origin, code_idx))

                        #check similarity
                        similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                        #if change_rate > ch_rate:
                        if similarity < similarity_limit:
                            print('=============> break by similarity limit on add')
                            adv_x_test[visit_idx, code_idx]=0
                            added_codes.remove((visit_idx, origin, code_idx))
                            final_adv = adv_x_test
                            flag = 0
                            break


                        #check attack
                        if score_added < comparing_score:
                            print('=============> break by time limit on add')
                            adv_x_test[visit_idx, code_idx]=0
                            added_codes.remove((visit_idx, origin, code_idx))
                            final_adv = adv_x_test
                            flag=1
                            break

                    else:
                        adv_x_test[visit_idx, code_idx]=0


        final_score = pred_duration(model, adv_x_test).numpy()[0]
        final_adv = adv_x_test
        similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
        if final_score >= s_real:
            print('attack failed')
        else:
            final_score = pred_duration(model, final_adv).numpy()[0]
            #change_rate = 100 * (len(added_codes) + len(removed_codes)) / x_attacked_patient[:, 2:].sum()
            if verbose:
                print('Black-Box attack is done => patient pred-time orders has been attacked!')
                print(f'original pred-time orders: p1 :{comparing_score:.3f} < p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} > p2 :{final_score:.3f}')
                print(f'number of perturbations: {len(removed_codes) + len(added_codes)}')
                print(f'number of removed : {len(removed_codes) - len(replaced_codes)}')
                print(f'number of replaced: {len(replaced_codes)}')
                print(f'number of added: {len(added_codes)}')
                #print(f'change rate: {change_rate:.2f}%')
                print(f'similarity rate: {similarity:.4f}%')
                visit_of_change = np.array([item[0] for item in removed_codes] + [item[0] for item in added_codes])
                for i in range(num_visits):
                    print(f'number of changes in visit {i + 1} : {(visit_of_change == i).sum()} / {x_attacked_patient[i, 2:].sum()}')

    #increase
    else:
        for item in sorted_codes:

            visit_idx = item[0][0]
            label = item[0][1]
            code_idx = item[0][2]

            current_score = pred_duration(model, adv_x_test).numpy()[0]

            if label == 'remove':
                if adv_x_test[visit_idx, code_idx]==1:
                    adv_x_test[visit_idx, code_idx]=0
                    score_removed = pred_duration(model, adv_x_test).numpy()[0]

                    if score_removed > min(current_score, s_real):
                        removed_codes.append((visit_idx, code_idx))

                        #check similarity
                        similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                        #if change_rate > ch_rate:
                        if similarity < similarity_limit:
                            print('=============> break by similarity limit on removal')
                            adv_x_test[visit_idx, code_idx]=1
                            removed_codes.remove((visit_idx, code_idx))
                            score = current_score
                            #final_adv = adv_x_test
                            #flag = 0
                            #break
                        #check attack
                        else:
                            if score_removed > comparing_score:
                                final_adv = adv_x_test
                                flag = 1
                                break
                    else:
                        adv_x_test[visit_idx, code_idx]=1


                 # checking replacing
                #if code_idx >= 1126:
                synonym_set = item[0][3]
                #synonym_set = get_synonym_codes(code_idx, True, onto_vocab_rx, rxdx_codes)
                ##################### replacing
                chosen_syn=0
                maximum=0
                for synonym_code in synonym_set:
                    if synonym_code in rxdx_codes:
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx]=0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp > current_score) & (np.abs(s_temp-current_score) > maximum):
                            maximum = np.abs(s_temp-current_score)
                            chosen_syn = synonym_code

                if chosen_syn!=0:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn]]==0:
                        if adv_x_test[visit_idx, code_idx]==0:
                            already_zero=1
                        else:
                            already_zero=0

                        adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=1
                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score > max(s_real, current_score, score_removed):
                            replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                            if already_zero==0:
                                removed_codes.append((visit_idx, code_idx))

                            # check similarity
                            similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                            #if change_rate > ch_rate:
                            if similarity < similarity_limit:
                                print('========================> break by similarity limit on replace')
                                adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                                replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn]))
                                if already_zero==0:
                                    removed_codes.remove((visit_idx, code_idx))
                                    adv_x_test[visit_idx, code_idx]=1
                                final_adv = adv_x_test
                                flag=0
                                break

                            else:
                                #time check
                                if added_syn_score > comparing_score:
                                    final_adv = adv_x_test
                                    flag = 1
                                    break
                        elif (added_syn_score > max(s_real, current_score)) & (added_syn_score < score_removed):
                            adv_x_test[visit_idx, code_idx]=0
                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                        else:
                            if already_zero==1:
                                adv_x_test[visit_idx, code_idx]=0
                            else:
                                adv_x_test[visit_idx, code_idx]=1

                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0

            else: #label=='add'
                origin = item[0][3]
                if adv_x_test[visit_idx, code_idx]==0:
                    adv_x_test[visit_idx, code_idx]=1
                    score_added = pred_duration(model, adv_x_test).numpy()[0]

                    if score_added > max(current_score, s_real):
                        added_codes.append((visit_idx, origin, code_idx))

                        #check similarity
                        similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                        #if change_rate > ch_rate:
                        if similarity < similarity_limit:
                            print('=============> break by similarity limit on add')
                            adv_x_test[visit_idx, code_idx]=0
                            added_codes.remove((visit_idx, origin, code_idx))
                            final_adv = adv_x_test
                            flag = 0
                            break

                        #check attack
                        if score_added > comparing_score:
                            final_adv = adv_x_test
                            flag = 1
                            break
                    else:
                        adv_x_test[visit_idx, code_idx]=0


        final_score = pred_duration(model, adv_x_test).numpy()[0]
        final_adv = adv_x_test
        similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
        if final_score <= s_real:
            print('attack failed')
        else:
            final_score = pred_duration(model, final_adv).numpy()[0]
            #change_rate = 100 * (len(added_codes) + len(removed_codes)) / x_attacked_patient[:, 2:].sum()
            if verbose:
                print('Black-Box attack is done => patient pred-time orders has been attacked!')
                print(f'original pred-time orders: p1 :{comparing_score:.3f} > p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} < p2 :{final_score:.3f}')
                print(f'number of perturbations: {len(removed_codes) + len(added_codes)}')
                print(f'number of removed : {len(removed_codes) - len(replaced_codes)}')
                print(f'number of replaced: {len(replaced_codes)}')
                print(f'number of added: {len(added_codes)}')
                #print(f'change rate: {change_rate:.2f}%')
                print(f'similarity rate: {similarity:.4f}%')
                visit_of_change = np.array([item[0] for item in removed_codes] + [item[0] for item in added_codes])
                for i in range(num_visits):
                    print(f'number of changes in visit {i + 1} : {(visit_of_change == i).sum()} / {x_attacked_patient[i, 2:].sum()}')

    return final_adv, removed_codes, replaced_codes, added_codes, final_score, change_rate, s_real, flag

def Bbox_AdvAttack_SA_pair_censor0_new(model, x_attacked_patient, rxdx_codes, onto_rx, onto_dx, similarity_limit=0.80, increase=False, verbose=True):
    """attack function for a censored data; lowring the predicted time/risk by comparing it with an obsorved data, contingent on the change rate of less than x percent"""

    code_ctoi = {code:i for i, code in enumerate(rxdx_codes)}
    comparing_score = 0

    code_ds_dict, sorted_codes, s_real = patient_code_score2(model, x_attacked_patient, rxdx_codes[2:], increase=increase)

    num_visits = x_attacked_patient.shape[0]

    if (not increase) & (s_real < comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, None, None


    adv_x_test = x_attacked_patient.copy()
    removed_codes = []
    replaced_codes = []
    final_adv = 0

    for e, item in enumerate(sorted_codes):
        visit_idx = item[0][0]
        code_idx = item[0][1]


        current_score = pred_duration(model, adv_x_test).numpy()[0]
        adv_x_test[visit_idx, code_idx]=0
        score = pred_duration(model, adv_x_test).numpy()[0]

        if score < min(current_score, s_real):
            removed_codes.append((visit_idx, code_idx))

            #change_rate = 100 * (len(removed_codes) - 0.5*len(replaced_codes)) / x_attacked_patient[:, 2:].sum()
            similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)

            #if change_rate > ch_rate:
            if similarity < similarity_limit:
                print('=============> passing similarity_limit on removal, lets check replacing then!')
                adv_x_test[visit_idx, code_idx]=1
                removed_codes.remove((visit_idx, code_idx))
                #final_adv = adv_x_test
                score = current_score
                #break

        else:
            adv_x_test[visit_idx, code_idx]=1

         # checking replacing
        if code_idx >= 1126:
            synonym_set = get_synonym_codes(code_idx, True, onto_rx, rxdx_codes)
        else:
            synonym_set = get_synonym_codes(code_idx, False, onto_dx, rxdx_codes)


        chosen_syn=0
        maximum = 0
        for synonym_code in synonym_set:
            if synonym_code in rxdx_codes:
                adv_x_test_copy = adv_x_test.copy()
                adv_x_test_copy[visit_idx, code_idx]=0
                adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                if (s_temp < current_score) & (np.abs(s_temp-current_score) > maximum):
                    maximum = np.abs(s_temp-current_score)
                    chosen_syn = synonym_code
        if chosen_syn != 0:
            if adv_x_test[visit_idx, code_ctoi[chosen_syn]]==0:
                if adv_x_test[visit_idx, code_idx]==0:
                    already_zero=1
                else:
                    already_zero=0
                adv_x_test[visit_idx, code_idx]=0
                adv_x_test[visit_idx, code_ctoi[chosen_syn]]=1
                added_syn_score = pred_duration(model, adv_x_test).numpy()[0]

                if added_syn_score < min(s_real, current_score, score):
                    replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                    if already_zero==0:
                        removed_codes.append((visit_idx, code_idx))

                    similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                    #if change_rate > ch_rate:
                    if similarity < similarity_limit:
                        print('========================> break by similarity_limit on replace')
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                        replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn]))
                        if already_zero==0:
                            removed_codes.remove((visit_idx, code_idx))
                            adv_x_test[visit_idx, code_idx]=1
                        final_adv = adv_x_test
                        break

                elif (added_syn_score < min(s_real, current_score)) & (added_syn_score > score): #it means base code already removed
                    #adv_x_test[visit_idx, code_idx]=0
                    adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0

                else:
                    if already_zero==1:
                        adv_x_test[visit_idx, code_idx]=0
                    else:
                        adv_x_test[visit_idx, code_idx]=1
                    adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0


    final_adv = adv_x_test
    final_score = pred_duration(model, final_adv).numpy()[0]
    if final_score >= s_real:
        print('attack failed')
        flag=0
        similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
    else:
        flag=1

    ####################################################################
    #change_rate = 100 * (len(removed_codes) - 0.5*len(replaced_codes)) / x_attacked_patient[:, 2:].sum()
        similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
        if verbose:
            print('Black-Box attack is done => patient pred-time orders has been attacked!')
            print(f'original pred-time orders: p1 :{comparing_score:.3f} < p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} > p2 :{final_score:.3f}')
            print(f'number of perturbations: {len(removed_codes)}')
            print(f'number of removed: {len(removed_codes) - len(replaced_codes)}')
            print(f'number of replaced: {len(replaced_codes)}')
            #print(f'change rate: {change_rate:.2f}%')
            print(f'similarity rate: {similarity:.4f}%')
            visit_of_change= np.array([item[0] for item in removed_codes])
            for i in range(num_visits):
                print(f'number of changes in visit {i+1} : {(visit_of_change==i).sum()} / {x_attacked_patient[i, 2:].sum()}')


    return final_adv, removed_codes, replaced_codes, final_score, similarity, s_real, flag

def Bbox_AdvAttack_SA_pair_censored_new_new(model, x_attacked_patient, rxdx_codes, onto_rx, onto_dx, co_occurrence_prob_matrix, cooccrence_ratio=0.15, similarity_limit=0.80, increase=False, verbose=True):

    """flip the predicted time/risk of two patients based on only code score
    removing + replacing + adding
    search space is based on the scores of both existing codes and potential adding codes
    """

    code_ctoi = {code:i for i, code in enumerate(rxdx_codes)}
    comparing_score = 0

    num_visits = x_attacked_patient.shape[0]
    #the new sorting consider both
    code_ds_dict, sorted_codes, s_real = patient_code_score_new(model,
                                                            x_attacked_patient,
                                                            rxdx_codes[2:],
                                                            onto_rx,
                                                            onto_dx,
                                                            code_ctoi,
                                                            co_occurrence_prob_matrix,
                                                            p_cooccur=cooccrence_ratio,
                                                            increase=increase)
    if (not increase) & (s_real < comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, None, None, None

    num_visits = x_attacked_patient.shape[0]
    adv_x_test = x_attacked_patient.copy()
    removed_codes = []
    replaced_codes = []
    added_codes = []
    final_adv = 0

    if not increase:
        for item in sorted_codes:

            visit_idx = item[0][0]
            label = item[0][1]
            code_idx = item[0][2]

            current_score = pred_duration(model, adv_x_test).numpy()[0]

            if label == 'remove':
                if adv_x_test[visit_idx, code_idx]==1:
                    adv_x_test[visit_idx, code_idx]=0
                    score_removed = pred_duration(model, adv_x_test).numpy()[0]

                    if score_removed < min(current_score, s_real):
                        removed_codes.append((visit_idx, code_idx))
                        #change_rate = 100 * (len(removed_codes) - 0.5*len(replaced_codes)) / x_attacked_patient[:, 2:].sum()
                        similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)

                        #if change_rate > ch_rate:
                        if similarity < similarity_limit:
                            print('=====> passing similarity_limit on removal, lets check replacing then!')
                            adv_x_test[visit_idx, code_idx]=1
                            removed_codes.remove((visit_idx, code_idx))
                            #final_adv = adv_x_test
                            score = current_score
                            #break
                    else:
                        adv_x_test[visit_idx, code_idx]=1

                 # checking replacing
                #if code_idx >= 1126:
                synonym_set = item[0][3]
                #synonym_set = get_synonym_codes(code_idx, True, onto_vocab_rx, rxdx_codes)
                ##################### replacing
                chosen_syn=0
                maximum=0
                for synonym_code in synonym_set:
                    if synonym_code in rxdx_codes:
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx]=0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp < current_score) & (np.abs(s_temp-current_score) > maximum):
                            maximum = np.abs(s_temp-current_score)
                            chosen_syn = synonym_code
                if chosen_syn!=0:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn]]==0:
                        if adv_x_test[visit_idx, code_idx]==0:
                            already_zero=1
                        else:
                            already_zero=0

                        adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=1
                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score < min(s_real, current_score, score_removed):
                            replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                            if already_zero==0:
                                removed_codes.append((visit_idx, code_idx))

                            similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                            #if change_rate > ch_rate:
                            if similarity < similarity_limit:
                                print('========================> break by similarity_limit on replace')
                                adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                                replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn]))
                                if already_zero==0:
                                    removed_codes.remove((visit_idx, code_idx))
                                    adv_x_test[visit_idx, code_idx]=1
                                final_adv = adv_x_test
                                break
                        elif (added_syn_score < min(s_real, current_score)) & (added_syn_score > score_removed):
                            adv_x_test[visit_idx, code_idx]=0
                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                        else:
                            if already_zero==1:
                                adv_x_test[visit_idx, code_idx]=0
                            else:
                                adv_x_test[visit_idx, code_idx]=1

                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0

            else: #label=='add'
                origin = item[0][3]
                if adv_x_test[visit_idx, code_idx]==0:
                    adv_x_test[visit_idx, code_idx]=1
                    score_added = pred_duration(model, adv_x_test).numpy()[0]

                    if score_added < min(current_score, s_real):
                        added_codes.append((visit_idx, origin, code_idx))

                        #check similarity
                        similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                        #if change_rate > ch_rate:
                        if similarity < similarity_limit:
                            print('=============> break by similarity limit on add')
                            adv_x_test[visit_idx, code_idx]=0
                            added_codes.remove((visit_idx, origin, code_idx))
                            final_adv = adv_x_test
                            break
                    else:
                        adv_x_test[visit_idx, code_idx]=0

    final_adv = adv_x_test
    final_score = pred_duration(model, final_adv).numpy()[0]
    similarity = get_similarity(x_attacked_patient, adv_x_test , MyModel, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)

    if final_score >= s_real:
        print('attack failed')
        flag=0
    else:
        flag=1
        if verbose:
            print('Black-Box attack is done => patient pred-time orders has been attacked!')
            print(f'original pred-time orders: p1 :{comparing_score:.3f} < p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} > p2 :{final_score:.3f}')
            print(f'number of perturbations: {len(removed_codes) + len(added_codes)}')
            print(f'number of removed : {len(removed_codes) - len(replaced_codes)}')
            print(f'number of replaced: {len(replaced_codes)}')
            print(f'number of added: {len(added_codes)}')
            #print(f'change rate: {change_rate:.2f}%')
            print(f'similarity rate: {similarity:.4f}%')
            visit_of_change = np.array([item[0] for item in removed_codes] + [item[0] for item in added_codes])
            for i in range(num_visits):
                print(f'number of changes in visit {i + 1} : {(visit_of_change == i).sum()} / {x_attacked_patient[i, 2:].sum()}')

    return final_adv, removed_codes, replaced_codes, added_codes, final_score, change_rate, s_real, flag
###############################


def SurvAttack1(model, x_attacked_patient, comparing_time, rxdx_codes, onto_rx, onto_dx, co_occurrence_prob_matrix, encoder, encoder_gpu, dx_names, rx_names, ctoi_dx, ctoi_rx, device, cooccrence_ratio=0.15, similarity_limit=0.80, increase=False, verbose=True):
    """flip the predicted time/risk of two patients based on only code score
    removing + replacing + adding
    search space is based on the scores of both existing codes and potential adding codes
    """
    flag = 'n'
    code_ctoi = {code:i for i, code in enumerate(rxdx_codes)}
    comparing_score = comparing_time

    # the new sorting consider both
    code_ds_dict, sorted_codes, s_real = patient_code_score_new_ScoreSym1_gpu(model,
                                                                          encoder_gpu,
                                                                          x_attacked_patient,
                                                                          rxdx_codes[2:],
                                                                          onto_rx,
                                                                          onto_dx,
                                                                          code_ctoi,
                                                                          co_occurrence_prob_matrix,
                                                                          cooccrence_ratio,
                                                                          dx_names,
                                                                          rx_names,
                                                                          ctoi_dx,
                                                                          ctoi_rx,
                                                                         device,
                                                                          increase=increase)


    num_visits = x_attacked_patient.shape[0]

    if (not increase) & (s_real < comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, None, None, None

    if (increase) & (s_real > comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, None, None, None

    adv_x_test = x_attacked_patient.copy()
    removed_codes = []
    replaced_codes = []
    added_codes = []
    final_adv = 0

    if not increase:
        for item in sorted_codes:

            visit_idx = item[0][0]
            label = item[0][1]
            code_idx = item[0][2]

            current_score = pred_duration(model, adv_x_test).numpy()[0]

            if label == 'remove':
                sym_pass_remove=False
                if adv_x_test[visit_idx, code_idx]==1:
                    adv_x_test[visit_idx, code_idx]=0
                    score_removed = pred_duration(model, adv_x_test).numpy()[0]

                    if score_removed < min(current_score, s_real):
                        removed_codes.append((visit_idx, code_idx))

                        #check similarity
                        similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                        #if change_rate > ch_rate:
                        if similarity < similarity_limit:
                            print('=============> passing similarity limit on removal')
                            adv_x_test[visit_idx, code_idx]=1
                            removed_codes.remove((visit_idx, code_idx))
                            score = current_score
                            sym_pass_remove = True
                            #final_adv = adv_x_test
                            #flag = 0
                            #break

                        #check attack
                        else:
                            if score_removed < comparing_score:
                                print('=============> break by time limit on removal')
                                adv_x_test[visit_idx, code_idx]=1
                                removed_codes.remove((visit_idx, code_idx))
                                final_adv = adv_x_test
                                flag=1
                                break

                    else:
                        adv_x_test[visit_idx, code_idx]=1


                 # checking replacing
                #if code_idx >= 1126:
                synonym_set = item[0][3]
                #synonym_set = get_synonym_codes(code_idx, True, onto_vocab_rx, rxdx_codes)
                ##################### replacing
                chosen_syn=0
                maximum=0
                for synonym_code in synonym_set:
                    if (synonym_code in rxdx_codes)&(adv_x_test[visit_idx, code_ctoi[synonym_code]]==0):
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx]=0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp < current_score) & (np.abs(s_temp-current_score) > maximum):
                            maximum = np.abs(s_temp-current_score)
                            chosen_syn = synonym_code

                if chosen_syn==0:
                    if sym_pass_remove:
                        final_adv = adv_x_test
                        break
                else:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn]]==0:
                        if adv_x_test[visit_idx, code_idx]==0:
                            already_zero=1
                        else:
                            already_zero=0

                        adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=1
                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score < min(s_real, current_score, score_removed):
                            replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                            if already_zero==0:
                                removed_codes.append((visit_idx, code_idx))

                            # check similarity
                            similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                            #if change_rate > ch_rate:
                            if similarity < similarity_limit:
                                print('========================> break by similarity-limit on replace')
                                adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                                replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn]))
                                if already_zero==0:
                                    removed_codes.remove((visit_idx, code_idx))
                                    adv_x_test[visit_idx, code_idx]=1
                                final_adv = adv_x_test
                                flag=0
                                break

                            else:
                                if added_syn_score < comparing_score:
                                    print('========================> break by time limit on replace')
                                    adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                                    replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn]))
                                    if already_zero==0:
                                        removed_codes.remove((visit_idx, code_idx))
                                        adv_x_test[visit_idx, code_idx]=1
                                    final_adv = adv_x_test
                                    flag=1
                                    break


                        elif (added_syn_score < min(s_real, current_score)) & (added_syn_score > score_removed):

                            adv_x_test[visit_idx, code_idx]=0
                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                            if sym_pass_remove:
                                final_adv = adv_x_test
                                break
                        else:
                            if already_zero==1:
                                adv_x_test[visit_idx, code_idx]=0
                            else:
                                adv_x_test[visit_idx, code_idx]=1

                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                            if sym_pass_remove:
                                final_adv = adv_x_test
                                break

            else: #label=='add'
                origin = item[0][3]
                if adv_x_test[visit_idx, code_idx]==0:
                    adv_x_test[visit_idx, code_idx]=1
                    score_added = pred_duration(model, adv_x_test).numpy()[0]

                    if score_added < min(current_score, s_real):
                        added_codes.append((visit_idx, origin, code_idx))

                        #check similarity
                        similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                        #if change_rate > ch_rate:
                        if similarity < similarity_limit:
                            print('=============> break by similarity limit on add')
                            adv_x_test[visit_idx, code_idx]=0
                            added_codes.remove((visit_idx, origin, code_idx))
                            final_adv = adv_x_test
                            flag = 0
                            break


                        #check attack
                        if score_added < comparing_score:
                            print('=============> break by time limit on add')
                            adv_x_test[visit_idx, code_idx]=0
                            added_codes.remove((visit_idx, origin, code_idx))
                            final_adv = adv_x_test
                            flag=1
                            break

                    else:
                        adv_x_test[visit_idx, code_idx]=0


        final_score = pred_duration(model, adv_x_test).numpy()[0]
        final_adv = adv_x_test
        similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
        if final_score >= s_real:
            print('attack failed')
        else:
            final_score = pred_duration(model, final_adv).numpy()[0]
            #change_rate = 100 * (len(added_codes) + len(removed_codes)) / x_attacked_patient[:, 2:].sum()
            if verbose:
                print('Black-Box attack is done => patient pred-time orders has been attacked!')
                print(f'original pred-time orders: p1 :{comparing_score:.3f} < p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} > p2 :{final_score:.3f}')
                print(f'number of perturbations: {len(removed_codes) + len(added_codes)}')
                print(f'number of removed : {len(removed_codes) - len(replaced_codes)}')
                print(f'number of replaced: {len(replaced_codes)}')
                print(f'number of added: {len(added_codes)}')
                #print(f'change rate: {change_rate:.2f}%')
                print(f'similarity rate: {similarity:.4f}%')
                visit_of_change = np.array([item[0] for item in removed_codes] + [item[0] for item in added_codes])
                for i in range(num_visits):
                    print(f'number of changes in visit {i + 1} : {(visit_of_change == i).sum()} / {x_attacked_patient[i, 2:].sum()}')

    #increase
    else:
        for item in sorted_codes:

            visit_idx = item[0][0]
            label = item[0][1]
            code_idx = item[0][2]

            current_score = pred_duration(model, adv_x_test).numpy()[0]

            if label == 'remove':
                sym_pass_remove = False
                if adv_x_test[visit_idx, code_idx]==1:
                    adv_x_test[visit_idx, code_idx]=0
                    score_removed = pred_duration(model, adv_x_test).numpy()[0]

                    if score_removed > min(current_score, s_real):
                        removed_codes.append((visit_idx, code_idx))

                        #check similarity
                        similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                        #if change_rate > ch_rate:
                        if similarity < similarity_limit:
                            print('=============> break by similarity limit on removal')
                            adv_x_test[visit_idx, code_idx]=1
                            removed_codes.remove((visit_idx, code_idx))
                            score = current_score
                            sym_pass_remove = True
                            #final_adv = adv_x_test
                            #flag = 0
                            #break
                        #check attack
                        else:
                            if score_removed > comparing_score:
                                final_adv = adv_x_test
                                flag = 1
                                break
                    else:
                        adv_x_test[visit_idx, code_idx]=1


                 # checking replacing
                #if code_idx >= 1126:
                synonym_set = item[0][3]
                #synonym_set = get_synonym_codes(code_idx, True, onto_vocab_rx, rxdx_codes)
                ##################### replacing
                chosen_syn=0
                maximum=0
                for synonym_code in synonym_set:
                    if (synonym_code in rxdx_codes)&(adv_x_test[visit_idx, code_ctoi[synonym_code]]==0):
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx]=0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp > current_score) & (np.abs(s_temp-current_score) > maximum):
                            maximum = np.abs(s_temp-current_score)
                            chosen_syn = synonym_code

                if chosen_syn!=0:
                    if sym_pass_remove:
                        final_adv = adv_x_test
                        break

                else:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn]]==0:
                        if adv_x_test[visit_idx, code_idx]==0:
                            already_zero=1
                        else:
                            already_zero=0

                        adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=1
                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score > max(s_real, current_score, score_removed):
                            replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                            if already_zero==0:
                                removed_codes.append((visit_idx, code_idx))

                            # check similarity
                            similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                            #if change_rate > ch_rate:
                            if similarity < similarity_limit:
                                print('========================> break by similarity limit on replace')
                                adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                                replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn]))
                                if already_zero==0:
                                    removed_codes.remove((visit_idx, code_idx))
                                    adv_x_test[visit_idx, code_idx]=1
                                final_adv = adv_x_test
                                flag=0
                                break

                            else:
                                #time check
                                if added_syn_score > comparing_score:
                                    final_adv = adv_x_test
                                    flag = 1
                                    break
                        elif (added_syn_score > max(s_real, current_score)) & (added_syn_score < score_removed):
                            adv_x_test[visit_idx, code_idx]=0
                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                            if sym_pass_remove:
                                final_adv = adv_x_test
                                break
                        else:
                            if already_zero==1:
                                adv_x_test[visit_idx, code_idx]=0
                            else:
                                adv_x_test[visit_idx, code_idx]=1

                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                            if sym_pass_remove:
                                final_adv = adv_x_test
                                break

            else: #label=='add'
                origin = item[0][3]
                if adv_x_test[visit_idx, code_idx]==0:
                    adv_x_test[visit_idx, code_idx]=1
                    score_added = pred_duration(model, adv_x_test).numpy()[0]

                    if score_added > max(current_score, s_real):
                        added_codes.append((visit_idx, origin, code_idx))

                        #check similarity
                        similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                        #if change_rate > ch_rate:
                        if similarity < similarity_limit:
                            print('=============> break by similarity limit on add')
                            adv_x_test[visit_idx, code_idx]=0
                            added_codes.remove((visit_idx, origin, code_idx))
                            final_adv = adv_x_test
                            flag = 0
                            break

                        #check attack
                        if score_added > comparing_score:
                            final_adv = adv_x_test
                            flag = 1
                            break
                    else:
                        adv_x_test[visit_idx, code_idx]=0


        final_score = pred_duration(model, adv_x_test).numpy()[0]
        final_adv = adv_x_test
        similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
        if final_score <= s_real:
            print('attack failed')
        else:
            final_score = pred_duration(model, final_adv).numpy()[0]
            #change_rate = 100 * (len(added_codes) + len(removed_codes)) / x_attacked_patient[:, 2:].sum()
            if verbose:
                print('Black-Box attack is done => patient pred-time orders has been attacked!')
                print(f'original pred-time orders: p1 :{comparing_score:.3f} > p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} < p2 :{final_score:.3f}')
                print(f'number of perturbations: {len(removed_codes) + len(added_codes)}')
                print(f'number of removed : {len(removed_codes) - len(replaced_codes)}')
                print(f'number of replaced: {len(replaced_codes)}')
                print(f'number of added: {len(added_codes)}')
                #print(f'change rate: {change_rate:.2f}%')
                print(f'similarity rate: {similarity:.4f}%')
                visit_of_change = np.array([item[0] for item in removed_codes] + [item[0] for item in added_codes])
                for i in range(num_visits):
                    print(f'number of changes in visit {i + 1} : {(visit_of_change == i).sum()} / {x_attacked_patient[i, 2:].sum()}')

    return final_adv, removed_codes, replaced_codes, added_codes, final_score, change_rate, s_real, flag

def SurvAttack_censored1(model, x_attacked_patient, rxdx_codes, onto_rx, onto_dx, co_occurrence_prob_matrix, encoder, encoder_gpu, dx_names, rx_names, ctoi_dx, ctoi_rx, device, cooccrence_ratio=0.15, similarity_limit=0.80, increase=False, verbose=True):

    code_ctoi = {code:i for i, code in enumerate(rxdx_codes)}
    comparing_score = 0

    num_visits = x_attacked_patient.shape[0]
    #the new sorting consider both
    code_ds_dict, sorted_codes, s_real = patient_code_score_new_ScoreSym1_gpu(model,
                                                                          encoder_gpu,
                                                                          x_attacked_patient,
                                                                          rxdx_codes[2:],
                                                                          onto_rx,
                                                                          onto_dx,
                                                                          code_ctoi,
                                                                          co_occurrence_prob_matrix,
                                                                          cooccrence_ratio,
                                                                          dx_names,
                                                                          rx_names,
                                                                          ctoi_dx,
                                                                          ctoi_rx,
                                                                          device,
                                                                          increase=increase)

    if (not increase) & (s_real < comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, None, None, None

    num_visits = x_attacked_patient.shape[0]
    adv_x_test = x_attacked_patient.copy()
    removed_codes = []
    replaced_codes = []
    added_codes = []
    final_adv = 0

    if not increase:
        for item in sorted_codes:

            visit_idx = item[0][0]
            label = item[0][1]
            code_idx = item[0][2]

            current_score = pred_duration(model, adv_x_test).numpy()[0]

            if label == 'remove':
                sym_pass_remove=False
                if adv_x_test[visit_idx, code_idx]==1:
                    adv_x_test[visit_idx, code_idx]=0
                    score_removed = pred_duration(model, adv_x_test).numpy()[0]

                    if score_removed < min(current_score, s_real):
                        removed_codes.append((visit_idx, code_idx))

                        #check similarity
                        similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                        #if change_rate > ch_rate:
                        if similarity < similarity_limit:
                            print('=============> passing similarity limit on removal')
                            adv_x_test[visit_idx, code_idx]=1
                            removed_codes.remove((visit_idx, code_idx))
                            score = current_score
                            sym_pass_remove = True
                            #final_adv = adv_x_test
                            #flag = 0
                            #break

                        #check attack
                        else:
                            if score_removed < comparing_score:
                                print('=============> break by time limit on removal')
                                adv_x_test[visit_idx, code_idx]=1
                                removed_codes.remove((visit_idx, code_idx))
                                final_adv = adv_x_test
                                flag=1
                                break

                    else:
                        adv_x_test[visit_idx, code_idx]=1


                 # checking replacing
                #if code_idx >= 1126:
                synonym_set = item[0][3]
                #synonym_set = get_synonym_codes(code_idx, True, onto_vocab_rx, rxdx_codes)
                ##################### replacing
                chosen_syn=0
                maximum=0
                for synonym_code in synonym_set:
                    if (synonym_code in rxdx_codes)&(adv_x_test[visit_idx, code_ctoi[synonym_code]]==0):
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx]=0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp < current_score) & (np.abs(s_temp-current_score) > maximum):
                            maximum = np.abs(s_temp-current_score)
                            chosen_syn = synonym_code

                if chosen_syn==0:
                    if sym_pass_remove:
                        final_adv = adv_x_test
                        break
                else:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn]]==0:
                        if adv_x_test[visit_idx, code_idx]==0:
                            already_zero=1
                        else:
                            already_zero=0

                        adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=1
                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score < min(s_real, current_score, score_removed):
                            replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                            if already_zero==0:
                                removed_codes.append((visit_idx, code_idx))

                            # check similarity
                            similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                            #if change_rate > ch_rate:
                            if similarity < similarity_limit:
                                print('========================> break by similarity-limit on replace')
                                adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                                replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn]))
                                if already_zero==0:
                                    removed_codes.remove((visit_idx, code_idx))
                                    adv_x_test[visit_idx, code_idx]=1
                                final_adv = adv_x_test
                                flag=0
                                break

                            else:
                                if added_syn_score < comparing_score:
                                    print('========================> break by time limit on replace')
                                    adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                                    replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn]))
                                    if already_zero==0:
                                        removed_codes.remove((visit_idx, code_idx))
                                        adv_x_test[visit_idx, code_idx]=1
                                    final_adv = adv_x_test
                                    flag=1
                                    break


                        elif (added_syn_score < min(s_real, current_score)) & (added_syn_score > score_removed):

                            adv_x_test[visit_idx, code_idx]=0
                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                            if sym_pass_remove:
                                final_adv = adv_x_test
                                break
                        else:
                            if already_zero==1:
                                adv_x_test[visit_idx, code_idx]=0
                            else:
                                adv_x_test[visit_idx, code_idx]=1

                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                            if sym_pass_remove:
                                final_adv = adv_x_test
                                break

            else: #label=='add'
                origin = item[0][3]
                if adv_x_test[visit_idx, code_idx]==0:
                    adv_x_test[visit_idx, code_idx]=1
                    score_added = pred_duration(model, adv_x_test).numpy()[0]

                    if score_added < min(current_score, s_real):
                        added_codes.append((visit_idx, origin, code_idx))

                        #check similarity
                        similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                        #if change_rate > ch_rate:
                        if similarity < similarity_limit:
                            print('=============> break by similarity limit on add')
                            adv_x_test[visit_idx, code_idx]=0
                            added_codes.remove((visit_idx, origin, code_idx))
                            final_adv = adv_x_test
                            flag = 0
                            break


                        #check attack
                        if score_added < comparing_score:
                            print('=============> break by time limit on add')
                            adv_x_test[visit_idx, code_idx]=0
                            added_codes.remove((visit_idx, origin, code_idx))
                            final_adv = adv_x_test
                            flag=1
                            break

                    else:
                        adv_x_test[visit_idx, code_idx]=0

    final_adv = adv_x_test
    final_score = pred_duration(model, final_adv).numpy()[0]
    similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)

    if final_score >= s_real:
        print('attack failed')
        flag=0
    else:
        flag=1
        if verbose:
            print('Black-Box attack is done => patient pred-time orders has been attacked!')
            print(f'original pred-time orders: p1 :{comparing_score:.3f} < p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} > p2 :{final_score:.3f}')
            print(f'number of perturbations: {len(removed_codes) + len(added_codes)}')
            print(f'number of removed : {len(removed_codes) - len(replaced_codes)}')
            print(f'number of replaced: {len(replaced_codes)}')
            print(f'number of added: {len(added_codes)}')
            #print(f'change rate: {change_rate:.2f}%')
            print(f'similarity rate: {similarity:.4f}%')
            visit_of_change = np.array([item[0] for item in removed_codes] + [item[0] for item in added_codes])
            for i in range(num_visits):
                print(f'number of changes in visit {i + 1} : {(visit_of_change == i).sum()} / {x_attacked_patient[i, 2:].sum()}')

    return final_adv, removed_codes, replaced_codes, added_codes, final_score, s_real,similarity, flag

def SurvAttack1_2(model, x_attacked_patient, comparing_time, rxdx_codes, onto_rx, onto_dx, co_occurrence_prob_matrix, encoder, encoder_gpu, dx_names, rx_names, ctoi_dx, ctoi_rx, device, cooccrence_ratio=0.50, similarity_limit=0.90, increase=False, verbose=True):
    """flip the predicted time/risk of two patients based on only code score
    removing + replacing + adding
    search space is based on the scores of both existing codes and potential adding codes
    """
    code_ctoi = {code:i for i, code in enumerate(rxdx_codes)}
    comparing_score = comparing_time

    # the new sorting consider both
    code_ds_dict, sorted_codes, s_real = patient_code_score_new_ScoreSym1_gpu2(model,
                                                                          encoder_gpu,
                                                                          x_attacked_patient,
                                                                          rxdx_codes[2:],
                                                                          onto_rx,
                                                                          onto_dx,
                                                                          code_ctoi,
                                                                          co_occurrence_prob_matrix,
                                                                          cooccrence_ratio,
                                                                          dx_names,
                                                                          rx_names,
                                                                          ctoi_dx,
                                                                          ctoi_rx,
                                                                          device,
                                                                          increase=increase)


    num_visits = x_attacked_patient.shape[0]

    if (not increase) & (s_real < comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, None, None, None

    if (increase) & (s_real > comparing_score):
        print('The order is already wrong ----> No need to attack')
        return None, None, None, None, None, None, None, None

    adv_x_test = x_attacked_patient.copy()
    removed_codes = []
    replaced_codes = []
    added_codes = []
    final_adv = 0

    if not increase:
        for item in sorted_codes:

            visit_idx = item[0][0]
            label = item[0][1]
            code_idx = item[0][2]

            current_score = pred_duration(model, adv_x_test).numpy()[0]

            if label == 'remove':
                sym_pass_remove=False
                if adv_x_test[visit_idx, code_idx]==1:
                    adv_x_test[visit_idx, code_idx]=0
                    score_removed = pred_duration(model, adv_x_test).numpy()[0]

                    if score_removed < min(current_score, s_real):
                        #check similarity
                        similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                        #if change_rate > ch_rate:
                        if similarity > similarity_limit:
                            removed_codes.append((visit_idx, code_idx))
                        #check attack
                            if score_removed < comparing_score:
                                print('=============> break by time limit on removal')
                                adv_x_test[visit_idx, code_idx]=1
                                removed_codes.remove((visit_idx, code_idx))
                                break
                        else:
                            print('=============> passing similarity limit on removal, lets try replace then!')
                            adv_x_test[visit_idx, code_idx]=1
                            score = current_score
                            sym_pass_remove = True
                    else:
                        adv_x_test[visit_idx, code_idx]=1


                 # checking replacing
                #if code_idx >= 1126:
                synonym_set = item[0][3]
                #synonym_set = get_synonym_codes(code_idx, True, onto_vocab_rx, rxdx_codes)
                ##################### replacing
                chosen_syn=0
                maximum=0
                for synonym_code in synonym_set:
                    if (synonym_code in rxdx_codes)&(adv_x_test[visit_idx, code_ctoi[synonym_code]]==0):
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx]=0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp < current_score) & (np.abs(s_temp-current_score) > maximum):
                            maximum = np.abs(s_temp-current_score)
                            chosen_syn = synonym_code

                if chosen_syn==0:
                    if sym_pass_remove:
                        break
                else:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn]]==0:
                        if adv_x_test[visit_idx, code_idx]==0:
                            already_zero=1
                        else:
                            already_zero=0

                        adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=1
                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score < min(s_real, current_score, score_removed):
                            # check similarity
                            similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                            #if change_rate > ch_rate:
                            if similarity > similarity_limit:
                                replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                                if already_zero==0:
                                    removed_codes.append((visit_idx, code_idx))

                                if added_syn_score < comparing_score:
                                    print('========================> break by time limit on replace')
                                    adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                                    replaced_codes.remove((visit_idx, code_idx, code_ctoi[chosen_syn]))
                                    if already_zero==0:
                                        removed_codes.remove((visit_idx, code_idx))
                                        adv_x_test[visit_idx, code_idx]=1
                                    break
                            else:
                                print('========================> break by similarity-limit on replace')
                                adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                                if already_zero==0:
                                    adv_x_test[visit_idx, code_idx]=1
                                break
                        else:
                            if already_zero==1:
                                adv_x_test[visit_idx, code_idx]=0
                            else:
                                adv_x_test[visit_idx, code_idx]=1

                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                            if sym_pass_remove:
                                break

            else: #label=='add'
                origin = item[0][3]
                if adv_x_test[visit_idx, code_idx]==0:
                    adv_x_test[visit_idx, code_idx]=1
                    score_added = pred_duration(model, adv_x_test).numpy()[0]

                    if score_added < min(current_score, s_real):

                        #check similarity
                        similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                        #if change_rate > ch_rate:
                        if similarity > similarity_limit:
                            added_codes.append((visit_idx, origin, code_idx))
                        #check attack
                            if score_added < comparing_score:
                                print('=============> break by time limit on add')
                                adv_x_test[visit_idx, code_idx]=0
                                added_codes.remove((visit_idx, origin, code_idx))
                                break
                        else:
                            print('=============> break by similarity limit on add')
                            adv_x_test[visit_idx, code_idx]=0
                            break

                    else:
                        adv_x_test[visit_idx, code_idx]=0


        final_score = pred_duration(model, adv_x_test).numpy()[0]
        final_adv = adv_x_test
        similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
        if final_score >= s_real:
            print('attack failed')
            flag=0
        else:
            flag=1
            if verbose:
                print('Black-Box attack is done => patient pred-time orders has been attacked!')
                print(f'original pred-time orders: p1 :{comparing_score:.3f} < p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} > p2 :{final_score:.3f}')
                print(f'number of perturbations: {len(removed_codes) + len(added_codes)}')
                print(f'number of removed : {len(removed_codes) - len(replaced_codes)}')
                print(f'number of replaced: {len(replaced_codes)}')
                print(f'number of added: {len(added_codes)}')
                #print(f'change rate: {change_rate:.2f}%')
                print(f'similarity rate: {similarity:.4f}%')
                visit_of_change = np.array([item[0] for item in removed_codes] + [item[0] for item in added_codes])
                for i in range(num_visits):
                    print(f'number of changes in visit {i + 1} : {(visit_of_change == i).sum()} / {x_attacked_patient[i, 2:].sum()}')

    #increase
    else:
        for item in sorted_codes:

            visit_idx = item[0][0]
            label = item[0][1]
            code_idx = item[0][2]

            current_score = pred_duration(model, adv_x_test).numpy()[0]

            if label == 'remove':
                sym_pass_remove=False
                if adv_x_test[visit_idx, code_idx]==1:
                    adv_x_test[visit_idx, code_idx]=0
                    score_removed = pred_duration(model, adv_x_test).numpy()[0]

                    if score_removed > max(current_score, s_real):
                        #check similarity
                        similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                        #if change_rate > ch_rate:
                        if similarity > similarity_limit:
                            removed_codes.append((visit_idx, code_idx))
                        #check attack
                            if score_removed > comparing_score:
                                print('=============> break by time limit on removal')
                                break
                        else:
                            print('=============> passing similarity limit on removal, lets try replace then!')
                            adv_x_test[visit_idx, code_idx]=1
                            score = current_score
                            sym_pass_remove = True
                    else:
                        adv_x_test[visit_idx, code_idx]=1


                 # checking replacing
                #if code_idx >= 1126:
                synonym_set = item[0][3]
                #synonym_set = get_synonym_codes(code_idx, True, onto_vocab_rx, rxdx_codes)
                ##################### replacing
                chosen_syn=0
                maximum=0
                for synonym_code in synonym_set:
                    if (synonym_code in rxdx_codes)&(adv_x_test[visit_idx, code_ctoi[synonym_code]]==0):
                        adv_x_test_copy = adv_x_test.copy()
                        adv_x_test_copy[visit_idx, code_idx]=0
                        adv_x_test_copy[visit_idx, code_ctoi[synonym_code]]=1
                        s_temp = pred_duration(model, adv_x_test_copy).numpy()[0]
                        if (s_temp > current_score) & (np.abs(s_temp-current_score) > maximum):
                            maximum = np.abs(s_temp-current_score)
                            chosen_syn = synonym_code

                if chosen_syn==0:
                    if sym_pass_remove:
                        break
                else:
                    if adv_x_test[visit_idx, code_ctoi[chosen_syn]]==0:
                        if adv_x_test[visit_idx, code_idx]==0:
                            already_zero=1
                        else:
                            already_zero=0

                        adv_x_test[visit_idx, code_idx]=0
                        adv_x_test[visit_idx, code_ctoi[chosen_syn]]=1
                        added_syn_score = pred_duration(model, adv_x_test).numpy()[0]
                        if added_syn_score > max(s_real, current_score, score_removed):

                            # check similarity
                            similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)

                            if similarity > similarity_limit:
                                replaced_codes.append((visit_idx, code_idx, code_ctoi[chosen_syn]))
                                if already_zero==0:
                                    removed_codes.append((visit_idx, code_idx))

                                if added_syn_score > comparing_score:
                                    print('========================> break by time limit on replace')
                                    break
                            else:
                                print('========================> break by similarity-limit on replace')
                                adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                                if already_zero==0:
                                    adv_x_test[visit_idx, code_idx]=1
                                break
                        else:
                            if already_zero==1:
                                adv_x_test[visit_idx, code_idx]=0
                            else:
                                adv_x_test[visit_idx, code_idx]=1

                            adv_x_test[visit_idx, code_ctoi[chosen_syn]]=0
                            if sym_pass_remove:
                                break

            else: #label=='add'
                origin = item[0][3]
                if adv_x_test[visit_idx, code_idx]==0:
                    adv_x_test[visit_idx, code_idx]=1
                    score_added = pred_duration(model, adv_x_test).numpy()[0]

                    if score_added > max(current_score, s_real):

                        #check similarity
                        similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
                        #if change_rate > ch_rate:
                        if similarity > similarity_limit:
                            added_codes.append((visit_idx, origin, code_idx))
                        #check attack
                            if score_added > comparing_score:
                                print('=============> break by time limit on add')
                                break
                        else:
                            print('=============> break by similarity limit on add')
                            adv_x_test[visit_idx, code_idx]=0
                            break

                    else:
                        adv_x_test[visit_idx, code_idx]=0

        final_score = pred_duration(model, adv_x_test).numpy()[0]
        final_adv = adv_x_test
        similarity = get_similarity(x_attacked_patient, adv_x_test , encoder, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx)
        if final_score <= s_real:
            print('attack failed')
            flag=0
        else:
            flag=1
            if verbose:
                print('Black-Box attack is done => patient pred-time orders has been attacked!')
                print(f'original pred-time orders: p1 :{comparing_score:.3f} > p2 :{s_real:.3f} ---> attacked orders: p1 :{comparing_score:.3f} < p2 :{final_score:.3f}')
                print(f'number of perturbations: {len(removed_codes) + len(added_codes)}')
                print(f'number of removed : {len(removed_codes) - len(replaced_codes)}')
                print(f'number of replaced: {len(replaced_codes)}')
                print(f'number of added: {len(added_codes)}')
                #print(f'change rate: {change_rate:.2f}%')
                print(f'similarity rate: {similarity:.4f}%')
                visit_of_change = np.array([item[0] for item in removed_codes] + [item[0] for item in added_codes])
                for i in range(num_visits):
                    print(f'number of changes in visit {i + 1} : {(visit_of_change == i).sum()} / {x_attacked_patient[i, 2:].sum()}')

    return final_adv, removed_codes, replaced_codes, added_codes, final_score, similarity, s_real, flag

