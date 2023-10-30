import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils_SA import *
from utils_attack import *


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


