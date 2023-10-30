# some util functions
import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter
import torch.nn.functional as F


# C(t)-INDEX CALCULATION
def c_index(Prediction, Time_survival, Death, Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    '''
    N = len(Prediction)
    A = np.zeros((N, N))
    Q = np.zeros((N, N))
    N_t = np.zeros((N, N))
    Num = 0
    Den = 0
    for i in range(N):
        A[i, np.where(Time_survival[i] < Time_survival)] = 1
        Q[i, np.where(Prediction[i] < Prediction)] = 1

        if (Time_survival[i] <= Time and Death[i] == 1):
            N_t[i, :] = 1

    Num = np.sum(((A) * N_t) * Q)
    Den = np.sum((A) * N_t)

    if Num == 0 and Den == 0:
        result = -1  # not able to compute c-index!
    else:
        result = float(Num / Den)

    return result


# BRIER-SCORE
def brier_score(Prediction, Time_survival, Death, Time):
    N = len(Prediction)
    y_true = ((Time_survival <= Time) * Death).astype(float)

    return np.mean((Prediction - y_true) ** 2)

    # result2[k, t] = brier_score_loss(risk[:, k], ((te_time[:,0] <= eval_horizon) * (te_label[:,0] == k+1)).astype(int))


#WEIGHTED C-INDEX & BRIER-SCORE
def CensoringProb(Y, T):
    T = T.reshape([-1])  # (N,) - np array
    Y = Y.reshape([-1])  # (N,) - np array

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=(Y == 0).astype(int))  # censoring prob = survival probability of event "censoring"
    G = np.asarray(kmf.survival_function_.reset_index()).transpose()
    G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]  # fill 0 with ZoH (to prevent nan values)

    return G


# C(t)-INDEX CALCULATION: this account for the weighted average for unbaised estimation
def weighted_c_index(T_train, Y_train, Prediction, T_test, Y_test, Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    '''
    G = CensoringProb(Y_train, T_train)

    N = len(Prediction)
    A = np.zeros((N, N))
    Q = np.zeros((N, N))
    N_t = np.zeros((N, N))
    Num = 0
    Den = 0
    for i in range(N):
        tmp_idx = np.where(G[0, :] >= T_test[i])[0]

        if len(tmp_idx) == 0:
            W = (1. / G[1, -1]) ** 2
        else:
            W = (1. / G[1, tmp_idx[0]]) ** 2

        A[i, np.where(T_test[i] < T_test)] = 1. * W
        Q[i, np.where(Prediction[i] < Prediction)] = 1.  # give weights

        if (T_test[i] <= Time and Y_test[i] == 1):
            N_t[i, :] = 1.

    Num = np.sum(((A) * N_t) * Q)
    Den = np.sum((A) * N_t)

    if Num == 0 and Den == 0:
        result = -1  # not able to compute c-index!
    else:
        result = float(Num / Den)

    return result


# this account for the weighted average for unbaised estimation
def weighted_brier_score(T_train, Y_train, Prediction, T_test, Y_test, Time):
    G = CensoringProb(Y_train, T_train)
    N = len(Prediction)

    W = np.zeros(len(Y_test))
    Y_tilde = (T_test > Time).astype(float)

    for i in range(N):
        tmp_idx1 = np.where(G[0, :] >= T_test[i])[0]
        tmp_idx2 = np.where(G[0, :] >= Time)[0]

        if len(tmp_idx1) == 0:
            G1 = G[1, -1]
        else:
            G1 = G[1, tmp_idx1[0]]

        if len(tmp_idx2) == 0:
            G2 = G[1, -1]
        else:
            G2 = G[1, tmp_idx2[0]]
        W[i] = (1. - Y_tilde[i]) * float(Y_test[i]) / G1 + Y_tilde[i] / G2

    y_true = ((T_test <= Time) * Y_test).astype(float)

    return np.mean(W * (Y_tilde - (1. - Prediction)) ** 2)


_EPSILON = 1e-08


import torch.nn.functional as F
def MAE_loss_single(pred_duration, trg_label, trg_ett):
    I = (trg_label == 1).float().squeeze()
    true_duration_ob = trg_ett[I == 1].squeeze()
    pred_duration_ob = pred_duration[I == 1]
    pred_duration_ob_i = pred_duration_ob + 1
    loss = F.l1_loss(pred_duration_ob_i, true_duration_ob)

    return loss

def MAE_loss(pred_duration, trg_label, trg_ett):

    I = (trg_label == 1).float().squeeze()
    true_duration_ob = trg_ett[I==1].squeeze()
    pred_duration_ob = pred_duration[I==1]
    pred_duration_ob_i = pred_duration_ob + 1
    loss = F.l1_loss(pred_duration_ob_i, true_duration_ob)

    return loss



# some util functions

_EPSILON = 1e-08
def div(x, y):
    return torch.div(x, (y + _EPSILON))

def log(x):
    return torch.log(x + _EPSILON)

def f_get_fc_mask(time, label, num_Event, num_Category):

    N = np.shape(time)[0]
    mask = np.ones([N, num_Event, num_Category])
    for i in range(N):
        if label[i] != 0:
            mask[i,int(label[i]-1), int(time[i])-1:] = 0
        else:
            mask[i,:,int(time[i]):] =  0

    return mask


def balance_by_increase(dataset, ratio):

    onset0 = dataset[dataset['event']==0]['onset'].unique()
    train_set_onset0 = dataset[dataset['event']==0].values

    onset1 = dataset[dataset['event'].isin([1,2])]['onset'].unique()
    train_set_onset1 = dataset[dataset['event'].isin([1,2])].values


    lst1 = [train_set_onset1 for i in range(ratio)]
    val1 = np.concatenate(lst1, axis=0)

    lst_on1 = []
    for i in range(ratio):
        lst_on1.append(onset1 + 0.01*i)

    on1 = np.concatenate(lst_on1, axis=0)

    lst_seq=[]
    for on in on1:
        lst_seq.extend([on for i in range(5)])


    increased_train_set_onset1 = pd.DataFrame(val1, columns=dataset.columns)
    increased_train_set_onset1['onset']=lst_seq

    increased_train_data_df = pd.concat([increased_train_set_onset1, dataset[dataset['event']==0]], axis=0)
    increased_train_data_df = increased_train_data_df.sort_values(['onset', 'time'])

    return increased_train_data_df


def balance_by_increase2(added_time_data, ratio):

    onset0 = added_time_data[added_time_data['event']==0]['onset'].unique()
    added_time_data_onset0 = added_time_data[added_time_data['event']==0].values
    onset1 = added_time_data[added_time_data['event'].isin([1,2])]['onset'].unique()
    added_time_data_onset1 = added_time_data[added_time_data['event'].isin([1,2])].values


    lst1 = [added_time_data_onset1 for i in range(ratio)]
    val1 = np.concatenate(lst1, axis=0)

    lst_on1 = []
    for i in range(ratio):
        lst_on1.append(onset1 + 0.01*i)

    on1 = np.concatenate(lst_on1, axis=0)


    increased_added_time_data_onset1 = pd.DataFrame(val1, columns=added_time_data.columns)
    increased_added_time_data_onset1['onset']=on1

    increased_added_time_data = pd.concat([increased_added_time_data_onset1, added_time_data[added_time_data['event']==0]], axis=0)
    increased_added_time_data = increased_added_time_data.sort_values(['onset'])

    return increased_added_time_data

def OrdinalRegLoss(surv_probs, target_label, mask):

    num_event = surv_probs.shape[1]

    neg_likelihood_loss = 0

    for i in range(num_event):
        predicted_surv_ = surv_probs[:,i,:]

        I_2 = (target_label == i + 1).float().squeeze()


        # for obsorved
        temp = torch.sum(log(mask[:,i,:] * predicted_surv_), dim=1) + torch.sum(log(1 - (1.0 - mask[:,i,:]) * predicted_surv_), dim=1)
        L_obsorved = I_2 * temp

        # for censored
        tmp2 = torch.sum(log(mask[:,i,:] * predicted_surv_), dim=1)
        L_censored = (1. - I_2) * tmp2

        neg_likelihood_loss += - (L_obsorved + L_censored)

    return torch.sum(neg_likelihood_loss)


def get_dx_vectors(onto_dx, v, max_length, pad_idx, dx_names, ctoi_dx):

    lst_l1=[]
    lst_l2=[]
    lst_l3=[]

    for code_l3 in v.nonzero():
        code_l1, code_l2, code_l3 = onto_dx[onto_dx['l3']==dx_names[code_l3.item()]][['l1', 'l2', 'l3']].values.squeeze()
        lst_l1.append(ctoi_dx[0][code_l1])
        lst_l2.append(ctoi_dx[1][code_l2])
        lst_l3.append(ctoi_dx[2][code_l3])

    padded_l1 = torch.full(size=(max_length,), fill_value=pad_idx)
    padded_l1[0:len(lst_l1)]= torch.tensor(lst_l1)


    padded_l2 = torch.full(size=(max_length,), fill_value=pad_idx)
    padded_l2[0:len(lst_l2)]= torch.tensor(lst_l2)

    padded_l3 = torch.full(size=(max_length,), fill_value=pad_idx)
    padded_l3[0:len(lst_l3)]= torch.tensor(lst_l3)

    return padded_l1, padded_l2, padded_l3


def get_rx_vectors(onto_rx, v, max_length, pad_idx, rx_names, ctoi_rx):

    lst_l1=[]
    lst_l2=[]
    lst_l3=[]
    lst_l4 = []
    for code_l4 in v.nonzero():
        code_l1, code_l2, code_l3, code_l4 = onto_rx[onto_rx['atc4']==rx_names[code_l4.item()]][['atc1', 'atc2', 'atc3', 'atc4']].values.squeeze()
        lst_l1.append(ctoi_rx[0][code_l1])
        lst_l2.append(ctoi_rx[1][code_l2])
        lst_l3.append(ctoi_rx[2][code_l3])
        lst_l4.append(ctoi_rx[3][code_l4])

    padded_l1 = torch.full(size=(max_length,), fill_value=pad_idx)
    padded_l1[0:len(lst_l1)]= torch.tensor(lst_l1)

    padded_l2 = torch.full(size=(max_length,), fill_value=pad_idx)
    padded_l2[0:len(lst_l2)]= torch.tensor(lst_l2)

    padded_l3 = torch.full(size=(max_length,), fill_value=pad_idx)
    padded_l3[0:len(lst_l3)]= torch.tensor(lst_l3)

    padded_l4 = torch.full(size=(max_length,), fill_value=pad_idx)
    padded_l4[0:len(lst_l4)]= torch.tensor(lst_l4)

    return padded_l1, padded_l2, padded_l3, padded_l4


def get_dx_levels(x_dx, onto_dx, dx_names, ctoi_dx, max_len=125, pad_token=0):
    l1=[]
    l2=[]
    l3=[]

    for row in x_dx:
        v1, v2, v3 = get_dx_vectors(onto_dx, row, max_len, pad_token, dx_names, ctoi_dx)
        l1.append(v1.unsqueeze(0))
        l2.append(v2.unsqueeze(0))
        l3.append(v3.unsqueeze(0))

    x_dx_l1 = torch.concat(l1, axis=0)
    x_dx_l2 = torch.concat(l2, axis=0)
    x_dx_l3 = torch.concat(l3, axis=0)

    x_dx_l1 = x_dx_l1.reshape(-1,5,x_dx_l1.shape[-1])
    x_dx_l2 = x_dx_l2.reshape(-1,5,x_dx_l2.shape[-1])
    x_dx_l3 = x_dx_l3.reshape(-1,5,x_dx_l3.shape[-1])

    return x_dx_l1, x_dx_l2, x_dx_l3

def get_rx_levels(x_rx, onto_rx, rx_names, ctoi_rx, max_len=125, pad_token=0):
    l1=[]
    l2=[]
    l3=[]
    l4=[]

    #get_rx_levels(onto_rx, x_rx, 100, 0)
    for row in x_rx:

        v1, v2, v3, v4 = get_rx_vectors(onto_rx, row, max_len, pad_token, rx_names, ctoi_rx)
        l1.append(v1.unsqueeze(0))
        l2.append(v2.unsqueeze(0))
        l3.append(v3.unsqueeze(0))
        l4.append(v4.unsqueeze(0))


    x_rx_l1 = torch.concat(l1, axis=0)
    x_rx_l2 = torch.concat(l2, axis=0)
    x_rx_l3 = torch.concat(l3, axis=0)
    x_rx_l4 = torch.concat(l4, axis=0)


    x_rx_l1 = x_rx_l1.reshape(-1,5,x_rx_l1.shape[-1])
    x_rx_l2 = x_rx_l2.reshape(-1,5,x_rx_l2.shape[-1])
    x_rx_l3 = x_rx_l3.reshape(-1,5,x_rx_l3.shape[-1])
    x_rx_l4 = x_rx_l4.reshape(-1,5,x_rx_l4.shape[-1])

    return x_rx_l1, x_rx_l2, x_rx_l3, x_rx_l4

def convert_onehot_to_idx(features, onto_dx, onto_rx, dx_names, rx_names, ctoi_dx, ctoi_rx, max_len=125, pad_token=0):
    features_dx = torch.tensor(features[:, :, 2:2+1346-3-219].astype('float32'))
    features_rx = torch.tensor(features[:, :, 2+1346-3-219:].astype('float32'))

    x_dx = features_dx.reshape(-1, features_dx.shape[-1])
    x_rx = features_rx.reshape(-1, features_rx.shape[-1])

    x_dx_l1, x_dx_l2, x_dx_l3 = get_dx_levels(x_dx, onto_dx, dx_names, ctoi_dx, max_len, pad_token)
    x_rx_l1, x_rx_l2, x_rx_l3, x_rx_l4 = get_rx_levels(x_rx, onto_rx, rx_names, ctoi_rx, max_len, pad_token)

    x_dx = (x_dx_l1, x_dx_l2, x_dx_l3)
    x_rx = (x_rx_l1, x_rx_l2, x_rx_l3, x_rx_l4)

    return x_dx, x_rx