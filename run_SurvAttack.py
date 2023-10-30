import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import date
import datetime
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import math
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_ipcw, brier_score
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve
from torch.nn import MSELoss
from sklearn.metrics.pairwise import cosine_similarity
from Encoder import *
from utils_SA import *
from utils_attack import *
from pycox.datasets import metabric
from pycox.models import LogisticHazard
from pycox.models import PMF
from pycox.models import MTLR
import pycox.models as models
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
import torchtuples as tt # Some useful functions
np.random.seed(1234)
_ = torch.manual_seed(123)



# importing vocabularies/ontology and doing preprocessing
path = 'data_samples/'
onto_vocab_rx = pd.read_csv(path+'ontovocab_prescription.csv')
onto_vocab_rx.drop(['Unnamed: 0'], axis=1, inplace=True)

onto_vocab_dx = pd.read_csv(path+'ontovocab_diagnosis.csv')
onto_vocab_dx.drop(['Unnamed: 0'], axis=1, inplace=True)

l_dx = np.load(path+'l_dx.npy')
l_rx = np.load(path+'l_rx.npy')

droped_icd9_codes = onto_vocab_dx[onto_vocab_dx['l1']=='UNK']['icd9'].unique()
droped_rxnorm_codes = onto_vocab_rx[onto_vocab_rx['atc1']=='UNK']['rx'].unique()
onto_vocab_rx_new = onto_vocab_rx.copy().drop(onto_vocab_rx[onto_vocab_rx['rx'].isin(droped_rxnorm_codes)].index)


onto_vocab_dx_new = onto_vocab_dx.drop(['l5', 'l4'], axis=1)
onto_vocab_rx_new = onto_vocab_rx_new.drop(['atc5'], axis=1)

onto_vocab_dx_new = onto_vocab_dx_new[~ onto_vocab_dx_new['l3'].isin(l_dx)]
onto_vocab_rx_new = onto_vocab_rx_new[~ onto_vocab_rx_new['atc4'].isin(l_rx)]


######################################## DX
onto_dx = onto_vocab_dx_new.drop_duplicates(subset='l3', keep='first')[['l1', 'l2', 'l3']]
onto_dx_l1 = onto_dx['l1'].unique()
n_onto_dx_l1 = onto_dx['l1'].nunique()

onto_dx_l2 = onto_dx['l2'].unique()
n_onto_dx_l2 = onto_dx['l2'].nunique()

onto_dx_l3 = onto_dx['l3'].unique()
n_onto_dx_l3 = onto_dx['l3'].nunique()


######################################## RX
onto_rx = onto_vocab_rx_new.drop_duplicates(subset='atc4', keep='first')[['atc1', 'atc2', 'atc3', 'atc4']]
onto_rx_l1 = onto_rx['atc1'].unique()
n_onto_rx_l1 = onto_rx['atc1'].nunique()


onto_rx_l2 = onto_rx['atc2'].unique()
n_onto_rx_l2 = onto_rx['atc2'].nunique()


onto_rx_l3 = onto_rx['atc3'].unique()
n_onto_rx_l3 = onto_rx['atc3'].nunique()


onto_rx_l4 = onto_rx['atc4'].unique()
n_onto_rx_l4 = onto_rx['atc4'].nunique()


onto_dx_params={'l1':n_onto_dx_l1+1, 'l2':n_onto_dx_l2+1, 'l3':n_onto_dx_l3+1}
onto_rx_params={'l1':n_onto_rx_l1+1, 'l2':n_onto_rx_l2+1, 'l3':n_onto_rx_l3+1, 'l4':n_onto_rx_l4+1}

#############################Creating Convertable Dictionaries
ctoi_dx1 = {code:idx+1 for idx, code in enumerate(onto_dx_l1)}
ctoi_dx1['<pad>'] = 0
itoc_dx1 = {idx+1:code for idx, code in enumerate(onto_dx_l1)}
itoc_dx1[0] = '<pad>'

ctoi_dx2 = {code:idx+1 for idx, code in enumerate(onto_dx_l2)}
ctoi_dx2['<pad>'] = 0
itoc_dx2 = {idx+1:code for idx, code in enumerate(onto_dx_l2)}
itoc_dx2[0] = '<pad>'

ctoi_dx3 = {code:idx+1 for idx, code in enumerate(onto_dx_l3)}
ctoi_dx3['<pad>'] = 0
itoc_dx3 = {idx+1:code for idx, code in enumerate(onto_dx_l3)}
itoc_dx3[0] = '<pad>'

ctoi_dx = [ctoi_dx1, ctoi_dx2, ctoi_dx3]


ctoi_rx1 = {code:idx+1 for idx, code in enumerate(onto_rx_l1)}
ctoi_rx1['<pad>'] = 0
itoc_rx1 = {idx+1:code for idx, code in enumerate(onto_rx_l1)}
itoc_rx1[0] = '<pad>'

ctoi_rx2 = {code:idx+1 for idx, code in enumerate(onto_rx_l2)}
ctoi_rx2['<pad>'] = 0
itoc_rx2 = {idx+1:code for idx, code in enumerate(onto_rx_l2)}
itoc_rx2[0] = '<pad>'

ctoi_rx3 = {code:idx+1 for idx, code in enumerate(onto_rx_l3)}
ctoi_rx3['<pad>'] = 0
itoc_rx3 = {idx+1:code for idx, code in enumerate(onto_rx_l3)}
itoc_rx3[0] = '<pad>'

ctoi_rx4 = {code:idx+1 for idx, code in enumerate(onto_rx_l4)}
ctoi_rx4['<pad>'] = 0
itoc_rx4 = {idx+1:code for idx, code in enumerate(onto_rx_l4)}
itoc_rx4[0] = '<pad>'

ctoi_rx = [ctoi_rx1, ctoi_rx2, ctoi_rx3, ctoi_rx4]

#######################################################################################

#loading encoder

#hyperparameters
encoder_num_layers = 2
code_dim=128
num_features = code_dim
embed_dim=512
rep_dim = 256
hidden_dim = 128
num_epochs = 30
learning_rate = 0.0001
num_event = 1



MyModel = Model(num_features=num_features,
                onto_dx_params=onto_dx_params,
                onto_rx_params=onto_rx_params,
                code_dim=code_dim,
                encoder_num_layers=1,
                embed_dim=embed_dim,
                rep_dim=rep_dim,
                hidden_dim=hidden_dim,
                num_event=num_event,
                num_category=1,
                device = 'cpu').to('cpu')



load_checkpoint(torch.load(path+"my_checkpoint_attack_encoder.pth.tar"), MyModel)


MyModel_gpu = Model(num_features=num_features,
                onto_dx_params=onto_dx_params,
                onto_rx_params=onto_rx_params,
                code_dim=code_dim,
                encoder_num_layers=1,
                embed_dim=embed_dim,
                rep_dim=rep_dim,
                hidden_dim=hidden_dim,
                num_event=num_event,
                num_category=1,
                device = 'cuda:0').to('cuda:0')


load_checkpoint(torch.load(path+"my_checkpoint_attack_encoder.pth.tar"), MyModel_gpu)

################################################

#importing AKI dataset

path = ''
train_set_df_=pd.read_csv(path+'final_data/train_set_df_.csv')
train_set_df_.drop(['Unnamed: 0'], axis=1, inplace=True)


added_time_df_tr_=pd.read_csv(path+'final_data/added_time_df_tr_.csv')
added_time_df_tr_.drop(['Unnamed: 0'], axis=1, inplace=True)


test_set_df=pd.read_csv(path+'final_data/test_set_df.csv')
test_set_df.drop(['Unnamed: 0'], axis=1, inplace=True)


added_time_df_test=pd.read_csv(path+'final_data/added_time_df_test.csv')
added_time_df_test.drop(['Unnamed: 0'], axis=1, inplace=True)

num_encounters = 5
num_category = 10
num_event=1

############################################ Dataset Vocabulary ########################################
class AKIDataset(Dataset):

    def __init__(self, df_dataset, num_sequence, num_category, num_event, added_time):

        self.df_dataset = df_dataset
        self.num_sequence = num_sequence
        self.added_time = added_time

        # formating data for model
        self.data_copy = self.df_dataset.values
        self.data_copy = self.data_copy.reshape(-1,self.num_sequence,self.df_dataset.shape[1])

        self.data_final_copy = self.data_copy[:,-1,:]
        self.time = self.data_final_copy[:, 4] #tte

        self.features = self.data_copy[:, :, 5:].astype('float32')
        self.day = self.data_copy[:, :, 2].astype('int16')
        self.tte = self.data_copy[:, :, 4].astype('float32')
        self.event = self.data_copy[:, :, 3].astype('int8')

        self.last_meas = self.data_final_copy[:, 2] # last measurement time
        self.last_meas = self.last_meas - self.added_time
        self.label = self.data_final_copy[:, 3] # event type

        self.num_category = num_category
        self.num_event = num_event

        self.mask3 = f_get_fc_mask(self.time, self.label, self.num_event, self.num_category)


    def __len__(self):
        return len(self.data_final_copy)

    def __getitem__(self, index):

        x = self.features[index]
        t = self.tte.reshape(-1, self.num_sequence,1)[index]
        y = self.event.reshape(-1, self.num_sequence,1)[index]
        day = self.day[index]
        m = self.mask3[index]

        return x, t, y, day, m



batch_size = 2000


akidataset_train = AKIDataset(df_dataset=train_set_df_,
                              num_sequence=5,
                              num_category=num_category,
                              num_event=num_event,
                              added_time=added_time_df_tr_['added_time'].values)



akidataset_test = AKIDataset(df_dataset=test_set_df,
                             num_sequence=5,
                             num_category=num_category,
                             num_event=num_event,
                             added_time=added_time_df_test['added_time'].values)



np.random.seed(1234)
_ = torch.manual_seed(123)

x_train = akidataset_train.features
x_train = x_train.reshape(x_train.shape[0], -1)
y_train = akidataset_train.event[:, -1]
#t_train = np.floor(akidataset_train.tte[:, -1])
t_train = akidataset_train.tte[:, -1]


x_test = akidataset_test.features
x_test = x_test.reshape(x_test.shape[0], -1)
y_test = akidataset_test.event[:, -1]
#t_test = np.floor(akidataset_test.tte[:, -1])
t_test = akidataset_test.tte[:, -1]

y_tr = (t_train, y_train)
y_te = (t_test, y_test)

train = (x_train, y_tr)
test = (x_test, y_te)

# Label transforms
num_durations = 10
labtrans = DeepHitSingle.label_transform(num_durations)
y_tr = labtrans.fit_transform(*y_tr)
y_te = labtrans.transform(*y_te)

train = (x_train, y_tr)
test = (x_test, y_te)


#load sa model
in_features = x_train.shape[1]
num_nodes = [512, 256]
out_features = labtrans.out_features
batch_norm = True
dropout = 0.2
net = torch.nn.Sequential(
     torch.nn.Linear(in_features, 512),
     torch.nn.ReLU(),
     torch.nn.BatchNorm1d(512),
     torch.nn.Dropout(0.1),

     torch.nn.Linear(512, 128),
     torch.nn.ReLU(),
     torch.nn.BatchNorm1d(128),
     torch.nn.Dropout(0.1),
     torch.nn.Linear(128, out_features))

model_DeepHitSingle = DeepHitSingle(net, tt.optim.Adam, alpha=0.1, sigma=0.1, duration_index=labtrans.cuts)
model_DeepHitSingle.load_model_weights("checkpoints/checkpoint_deephit.pth.tar")




# Attcking data preprocessing
import time

x_demo = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cooccrence_ratio = 0.50
change_rate = 15
co_occurrence_prob_matrix = np.load('saved_arrays/co_occurrence_prob_matrix.npy')
cooccrence_ratio = 0.50
dx_names = np.load('saved_arrays/dx_names.npy', allow_pickle=True)
rx_names = np.load('saved_arrays/rx_names.npy', allow_pickle=True)
rxdx_names = np.load('saved_arrays/rxdx_names.npy', allow_pickle=True)
x_test = np.load('saved_arrays/x_test.npy')
y_test = np.load('saved_arrays/y_test.npy')
t_test = np.load('saved_arrays/t_test.npy')


x_test_flat = x_test.reshape(x_test.shape[0], -1)
surv = model_DeepHitSingle.predict_surv_df(x_test_flat)
ev = EvalSurv(surv, t_test, y_test, censor_surv='km')
print('---c-index on whole data before attack---')
print(ev.concordance_td('antolini'))

x_test2 = x_test.copy()
t_test2 = t_test.copy()
y_test2 = y_test.copy()

y_test2_ob = y_test2[y_test2==1]
y_test2_cen = y_test2[y_test2==0]

x_test2_ob = x_test2[y_test2==1]
x_test2_cen = x_test2[y_test2==0]

t_test2_ob = t_test2[y_test2==1]
t_test2_cen = t_test2[y_test2==0]

print(f'number of observed data points: {y_test2_ob.shape[0]}')
print(f'number of cendored data points: {y_test2_cen.shape[0]}')

sorted_test_t_ob = sorted(zip(np.arange(len(t_test2_ob)), t_test2_ob), key= lambda x:x[1], reverse=True)
sorted_index_ob, sorted_t_test_ob = zip(*sorted_test_t_ob)
sorted_x_test_ob =  x_test2_ob[list(sorted_index_ob)]
sorted_y_test_ob =  y_test2_ob[list(sorted_index_ob)]

x_test2_ob_flat = x_test2_ob.reshape(x_test2_ob.shape[0], -1)
surv = model_DeepHitSingle.predict_surv_df(x_test2_ob_flat)
ev = EvalSurv(surv, t_test2_ob, y_test2_ob, censor_surv='km')
print('---c-index for only observed data before attack---')
print(ev.concordance_td('antolini'))


#evaluation -> mae
import torch.nn.functional as F
surv = model_DeepHitSingle.predict_surv_df(x_test2_ob_flat)
pred_duration_cc_ob = []
for i in range(surv.shape[1]):
    area = np.trapz(surv.values[:,i], x=surv.index)
    pred_duration_cc_ob.append(area)

print('mae', F.l1_loss(torch.tensor(pred_duration_cc_ob), torch.tensor(t_test2_ob)))



#start SurvAttack
adversarial_x_cen_deephit = np.zeros_like(x_test2_cen)
changed_code_dist_remove = np.zeros((x_test.shape[-2], x_test.shape[-1]))
changed_code_dist_replace = np.zeros((x_test.shape[-2], x_test.shape[-1]))
changed_code_dist_add = np.zeros((x_test.shape[-2], x_test.shape[-1]))
rxdx_codes = rxdx_names

for i in range(len(x_test2_cen)):
    print(f'\n-----------------------------about to attack censored instance {i} / {len(x_test2_cen)}\n')
    final_adv_cen, removed_codes_cen, replaced_codes_cen, added_codes_cen, final_score_cen, similarity_cen, s_real_cen, flag = Bbox_AdvAttack_SA_pair_censored_new_new_scoresim1(model_DeepHitSingle, x_test2_cen[i], rxdx_codes, onto_rx, onto_dx, co_occurrence_prob_matrix, MyModel, MyModel_gpu, dx_names, rx_names, ctoi_dx, ctoi_rx, device, cooccrence_ratio=0.5, similarity_limit=0.90, increase=False, verbose=True)
    adversarial_x_cen_deephit[i] = final_adv_cen

    if flag == 1:
        if len(removed_codes_cen)!=0:
            v, c = zip(*removed_codes_cen)
            for x, y in zip(v, c):
                changed_code_dist_remove[x, y] += 1

        if len(added_codes_cen)!=0:
            v, o, c = zip(*added_codes_cen)
            for x, y in zip(v, c):
                changed_code_dist_add[x, y] += 1

        if len(replaced_codes_cen)!=0:
            v, o, c = zip(*replaced_codes_cen)
            for x, y in zip(v, c):
                changed_code_dist_replace[x, y] += 1


print('c-index with only attacking censored data')
adversarial_x_cen_flat = adversarial_x_cen_deephit.reshape(adversarial_x_cen_deephit.shape[0], -1)
adv_x = np.concatenate([x_test2_ob_flat, adversarial_x_cen_flat], axis=0)
adv_t = np.concatenate([t_test2_ob, t_test2_cen], axis=0)
adv_y = np.concatenate([y_test2_ob, y_test2_cen], axis=0)
surv = model_DeepHitSingle.predict_surv_df(adv_x)
ev = EvalSurv(surv, adv_t, adv_y, censor_surv='km')
print(f'pycox c1^*-index: {ev.concordance_td("antolini")}')

#defining t_min
surv = model_DeepHitSingle.predict_surv_df(adversarial_x_cen_flat)
pred_duration_cen = []
for i in range(surv.shape[1]):
    area = np.trapz(surv.values[:,i], x=surv.index)
    pred_duration_cen.append(area)


t_min = np.array(pred_duration_cen).min()
rxdx_codes = rxdx_names
adversial_x_test_ob = np.zeros_like(sorted_x_test_ob)
changed_code_dist_replace_ob = np.zeros((x_test.shape[-2], x_test.shape[-1]))
changed_code_dist_remove_ob = np.zeros((x_test.shape[-2], x_test.shape[-1]))
changed_code_dist_add_ob = np.zeros((x_test.shape[-2], x_test.shape[-1]))
num_sucess_attacks=0
num_attacks=0
num_reduction=0
num_increase=0



for i in range(len(sorted_t_test_ob)):
    print()
    print(f'---------------------------------attacking patient {i} / {len(sorted_t_test_ob)}')
    original_pred_time = pred_duration(model_DeepHitSingle, sorted_x_test_ob[i]).numpy()[0]
    if original_pred_time > t_min:
        final_adv_ob, removed_codes_ob, replaced_codes_ob, added_codes_ob, final_score_ob, similarity_ob, s_real_ob, flag_ob = Bbox_AdvAttack_SA_pair_new_new_scoresim1_3(model_DeepHitSingle, sorted_x_test_ob[i], comparing_time=t_min, rxdx_codes=rxdx_codes, onto_rx=onto_rx, onto_dx=onto_dx, co_occurrence_prob_matrix=co_occurrence_prob_matrix, encoder=MyModel, encoder_gpu=MyModel_gpu, dx_names=dx_names, rx_names=rx_names, ctoi_dx=ctoi_dx, ctoi_rx=ctoi_rx, device=device, cooccrence_ratio=0.50, similarity_limit=0.90, increase=False, verbose=True)
        if final_score_ob != None:
            t_min = final_score_ob
        num_reduction+=1
    else:
        final_adv_ob, removed_codes_ob, replaced_codes_ob, added_codes_ob, final_score_ob, similarity_ob, s_real_ob, flag_ob = Bbox_AdvAttack_SA_pair_new_new_scoresim1_3(model_DeepHitSingle, sorted_x_test_ob[i], comparing_time=t_min, rxdx_codes=rxdx_codes, onto_rx=onto_rx, onto_dx=onto_dx, co_occurrence_prob_matrix=co_occurrence_prob_matrix, encoder=MyModel, encoder_gpu=MyModel_gpu, dx_names=dx_names, rx_names=rx_names, ctoi_dx=ctoi_dx, ctoi_rx=ctoi_rx, device=device, cooccrence_ratio=0.50, similarity_limit=0.90, increase=True, verbose=True)
        if final_score_ob != None:
            t_min = max(t_min, final_score_ob)
        num_increase+=1

    adversial_x_test_ob[i] = final_adv_ob
    if flag_ob == 1:
        if len(removed_codes_ob)!=0:
            v, c = zip(*removed_codes_ob)
            for x, y in zip(v, c):
                changed_code_dist_remove_ob[x, y] += 1

        if len(added_codes_ob)!=0:
            v, o, c = zip(*added_codes_ob)
            for x, y in zip(v, c):
                changed_code_dist_add_ob[x, y] += 1

        if len(replaced_codes_ob)!=0:
            v, o, c = zip(*replaced_codes_ob)
            for x, y in zip(v, c):
                changed_code_dist_replace_ob[x, y] += 1






print('c-index on only observed data before attack')
x_test2_ob_flat = x_test2_ob.reshape(x_test2_ob.shape[0], -1)
surv = model_DeepHitSingle.predict_surv_df(x_test2_ob_flat)
ev = EvalSurv(surv, t_test2_ob, y_test2_ob, censor_surv='km')
print('---c-index before attack---')
print(f'c_ob: {ev.concordance_td("antolini")}')
print()

print('c-index on only observed data after attack')
adversial_x_test_ob_flat = adversial_x_test_ob.reshape(adversial_x_test_ob.shape[0], -1)
surv = model_DeepHitSingle.predict_surv_df(adversial_x_test_ob_flat)
ev = EvalSurv(surv, np.array(sorted_t_test_ob), sorted_y_test_ob, censor_surv='km')
print(f'c_ob*: {ev.concordance_td("antolini")}')

print()
print('c-index with only attacking obsorved data')
x_test2_cen_flat = x_test2_cen.reshape(x_test2_cen.shape[0], -1)
adv_x = np.concatenate([adversial_x_test_ob_flat, x_test2_cen_flat], axis=0)
adv_t = np.concatenate([np.array(sorted_t_test_ob), t_test2_cen], axis=0)
adv_y = np.concatenate([sorted_y_test_ob, y_test2_cen], axis=0)
surv = model_DeepHitSingle.predict_surv_df(adv_x)
ev = EvalSurv(surv, adv_t, adv_y, censor_surv='km')
print(f'pycox c2*-index: {ev.concordance_td("antolini")}')

print()
print('c-index with attacking both observed and censored data')
adversarial_x_cen_flat = adversarial_x_cen_deephit.reshape(adversarial_x_cen_deephit.shape[0], -1)
adv_x = np.concatenate([adversial_x_test_ob_flat, adversarial_x_cen_flat], axis=0)
adv_t = np.concatenate([np.array(sorted_t_test_ob), t_test2_cen], axis=0)
adv_y = np.concatenate([sorted_y_test_ob, y_test2_cen], axis=0)
surv = model_DeepHitSingle.predict_surv_df(adv_x)
ev = EvalSurv(surv, adv_t, adv_y, censor_surv='km')
print(f'pycox c3*-index: {ev.concordance_td("antolini")}')