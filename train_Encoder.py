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
from utils_SA import *
from Encoder import *





# importing vocabularies/ontology and doing preprocessing
path = '/data_samples/'
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


######################################## DX ######################################################
onto_dx = onto_vocab_dx_new.drop_duplicates(subset='l3', keep='first')[['l1', 'l2', 'l3']]
onto_dx_l1 = onto_dx['l1'].unique()
n_onto_dx_l1 = onto_dx['l1'].nunique()

onto_dx_l2 = onto_dx['l2'].unique()
n_onto_dx_l2 = onto_dx['l2'].nunique()

onto_dx_l3 = onto_dx['l3'].unique()
n_onto_dx_l3 = onto_dx['l3'].nunique()


######################################## RX ######################################################
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

#############################Creating Convertable Dictionaries ##########################################


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



np.random.seed(1234)
_ = torch.manual_seed(123)


#data loading and preprocessing ##############################################
########### dx
x_dx_l1_d = pd.read_csv(path+'x_dx_l1_d_new.csv')
x_dx_l1_d.drop(['Unnamed: 0'], axis=1, inplace=True)
x_dx_l1_d = x_dx_l1_d.astype('int32')

x_dx_l2_d = pd.read_csv(path+'x_dx_l2_d_new.csv')
x_dx_l2_d.drop(['Unnamed: 0'], axis=1, inplace=True)
x_dx_l2_d = x_dx_l2_d.astype('int32')

x_dx_l3_d = pd.read_csv(path+'x_dx_l3_d_new.csv')
x_dx_l3_d.drop(['Unnamed: 0'], axis=1, inplace=True)
x_dx_l3_d = x_dx_l3_d.astype('int32')


############ rx
x_rx_l1_d = pd.read_csv(path+'x_rx_l1_d_new.csv')
x_rx_l1_d.drop(['Unnamed: 0'], axis=1, inplace=True)
x_rx_l1_d = x_rx_l1_d.astype('int32')

x_rx_l2_d = pd.read_csv(path+'x_rx_l2_d_new.csv')
x_rx_l2_d.drop(['Unnamed: 0'], axis=1, inplace=True)
x_rx_l2_d = x_rx_l2_d.astype('int32')

x_rx_l3_d = pd.read_csv(path+'x_rx_l3_d_new.csv')
x_rx_l3_d.drop(['Unnamed: 0'], axis=1, inplace=True)
x_rx_l3_d = x_rx_l3_d.astype('int32')

x_rx_l4_d = pd.read_csv(path+'x_rx_l4_d_new.csv')
x_rx_l4_d.drop(['Unnamed: 0'], axis=1, inplace=True)
x_rx_l4_d = x_rx_l4_d.astype('int32')

# importing dataset
dataset_df8 = pd.read_csv(path + 'Dataset3.csv')
dataset_df8.drop(['Unnamed: 0'], axis=1, inplace=True)


np.random.seed(1234)
_ = torch.manual_seed(123)

#creating added time dataset
added_time = dataset_df8.values.reshape((-1,5,dataset_df8.shape[1]))[:, 0, 2]
added_time_df = pd.DataFrame()
added_time_df['onset'] = dataset_df8['onset'].unique()
added_time_df['event'] = dataset_df8.values.reshape(-1,5,dataset_df8.shape[1])[:,-1,3]
added_time_df['added_time'] = added_time


### implementing added_time to dataset so the time will start from zero for all onsets
l = []
for time in added_time:
    l.extend(5 * [time])
dataset_df8['time'] = dataset_df8['time'].values - l


# merging all AKI 1 2 3, so we have only one event
dataset_df8['event'].replace([1.0,2.0, 3.0], 1.0, inplace=True)
added_time_df['event'].replace([1.0,2.0, 3.0], 1.0, inplace=True)
x_dx_l1_d['event'].replace([1.0,2.0, 3.0], 1.0, inplace=True)
x_dx_l2_d['event'].replace([1.0,2.0, 3.0], 1.0, inplace=True)
x_dx_l3_d['event'].replace([1.0,2.0, 3.0], 1.0, inplace=True)
x_rx_l1_d['event'].replace([1.0,2.0, 3.0], 1.0, inplace=True)
x_rx_l2_d['event'].replace([1.0,2.0, 3.0], 1.0, inplace=True)
x_rx_l3_d['event'].replace([1.0,2.0, 3.0], 1.0, inplace=True)
x_rx_l4_d['event'].replace([1.0,2.0, 3.0], 1.0, inplace=True)



# eliminating few onsets with very large time of aki so the num_category would be smaller
data_copy = dataset_df8.values
data_copy = data_copy.reshape(-1,5,dataset_df8.shape[1])
data_final_copy = data_copy[:,-1,:]
time = data_final_copy[:, 4]


plt.hist(np.floor(time), bins=100)
plt.xlabel('time to event')
plt.show()


print(len(time[time <= 10]) / len(time))
print('removing -------------------------')
print(len(time))
selected_onsets = dataset_df8['onset'].unique()[time <= 10]
print(len(selected_onsets))
dataset_df8 = dataset_df8[dataset_df8['onset'].isin(selected_onsets)]
print(dataset_df8['onset'].nunique())
data_copy = dataset_df8.values
data_copy = data_copy.reshape(-1,5,dataset_df8.shape[1])
data_final_copy = data_copy[:,-1,:]
time = data_final_copy[:, 4]

plt.hist(np.floor(time), bins=100)
plt.xlabel('time to event')
plt.show()


plt.hist(time, bins=100)
plt.xlabel('time to event')
plt.show()



added_time_df = added_time_df[added_time_df['onset'].isin(dataset_df8['onset'].unique())]

################## splitting Data to train/test

data_0_onset = dataset_df8[dataset_df8['event'] == 0]['onset'].unique()
data_1_onset = dataset_df8[dataset_df8['event'] == 1]['onset'].unique()


train_set_0, test_set_0 = train_test_split(data_0_onset, test_size=0.25, random_state=42)
train_set_1, test_set_1 = train_test_split(data_1_onset, test_size=0.25, random_state=42)


train_set_onsets = np.concatenate([train_set_0, train_set_1])
np.random.shuffle(train_set_onsets)
test_set_onsets = np.concatenate([test_set_0, test_set_1])
np.random.shuffle(test_set_onsets)


train_set_df = dataset_df8[dataset_df8['onset'].isin(train_set_onsets)].sort_values(['onset', 'time']).reset_index(drop=True)
x_dx_l1_d_train = x_dx_l1_d[x_dx_l1_d['onset'].isin(train_set_onsets)].sort_values(['onset', 'time']).reset_index(drop=True)
x_dx_l2_d_train = x_dx_l2_d[x_dx_l2_d['onset'].isin(train_set_onsets)].sort_values(['onset', 'time']).reset_index(drop=True)
x_dx_l3_d_train = x_dx_l3_d[x_dx_l3_d['onset'].isin(train_set_onsets)].sort_values(['onset', 'time']).reset_index(drop=True)
x_rx_l1_d_train = x_rx_l1_d[x_rx_l1_d['onset'].isin(train_set_onsets)].sort_values(['onset', 'time']).reset_index(drop=True)
x_rx_l2_d_train = x_rx_l2_d[x_rx_l2_d['onset'].isin(train_set_onsets)].sort_values(['onset', 'time']).reset_index(drop=True)
x_rx_l3_d_train = x_rx_l3_d[x_rx_l3_d['onset'].isin(train_set_onsets)].sort_values(['onset', 'time']).reset_index(drop=True)
x_rx_l4_d_train = x_rx_l4_d[x_rx_l4_d['onset'].isin(train_set_onsets)].sort_values(['onset', 'time']).reset_index(drop=True)



test_set_df = dataset_df8[dataset_df8['onset'].isin(test_set_onsets)].sort_values(['onset', 'time']).reset_index(drop=True)
x_dx_l1_d_test = x_dx_l1_d[x_dx_l1_d['onset'].isin(test_set_onsets)].sort_values(['onset', 'time']).reset_index(drop=True)
x_dx_l2_d_test = x_dx_l2_d[x_dx_l2_d['onset'].isin(test_set_onsets)].sort_values(['onset', 'time']).reset_index(drop=True)
x_dx_l3_d_test = x_dx_l3_d[x_dx_l3_d['onset'].isin(test_set_onsets)].sort_values(['onset', 'time']).reset_index(drop=True)
x_rx_l1_d_test = x_rx_l1_d[x_rx_l1_d['onset'].isin(test_set_onsets)].sort_values(['onset', 'time']).reset_index(drop=True)
x_rx_l2_d_test = x_rx_l2_d[x_rx_l2_d['onset'].isin(test_set_onsets)].sort_values(['onset', 'time']).reset_index(drop=True)
x_rx_l3_d_test = x_rx_l3_d[x_rx_l3_d['onset'].isin(test_set_onsets)].sort_values(['onset', 'time']).reset_index(drop=True)
x_rx_l4_d_test = x_rx_l4_d[x_rx_l4_d['onset'].isin(test_set_onsets)].sort_values(['onset', 'time']).reset_index(drop=True)



added_time_df_tr = added_time_df[added_time_df['onset'].isin(train_set_df['onset'].unique())].sort_values('onset')
added_time_df_test = added_time_df[added_time_df['onset'].isin(test_set_df['onset'].unique())].sort_values('onset')



########################################## formating data for model
num_encounters = 5
data_copy = dataset_df8.values
data_copy = data_copy.reshape(-1,num_encounters,dataset_df8.shape[1])
data_final_copy = data_copy[:,-1,:]
time = data_final_copy[:, 4]
label = data_final_copy[:, 3] # event type
num_category = int(time.max() * 1.1)
print(f'num_category : {num_category}')
num_event = len(np.unique(label)) - 1
print(f'num_event : {num_event}')


####### balancing train-data by increasgin : 50% censored / 50% obsorbed
ration = 14
train_set_df_ = balance_by_increase(train_set_df, ration)
x_dx_l1_d_train_ = balance_by_increase(x_dx_l1_d_train, ration)
x_dx_l2_d_train_ = balance_by_increase(x_dx_l2_d_train, ration)
x_dx_l3_d_train_ = balance_by_increase(x_dx_l3_d_train, ration)
x_rx_l1_d_train_ = balance_by_increase(x_rx_l1_d_train, ration)
x_rx_l2_d_train_ = balance_by_increase(x_rx_l2_d_train, ration)
x_rx_l3_d_train_ = balance_by_increase(x_rx_l3_d_train, ration)
x_rx_l4_d_train_ = balance_by_increase(x_rx_l4_d_train, ration)

added_time_df_tr_ = balance_by_increase2(added_time_df_tr, ration)

############################################ Dataset Vocabulary



class AKIDataset(Dataset):

    def __init__(self, df_dataset, x_dx, x_rx, num_sequence, num_category, num_event, added_time):

        self.df_dataset = df_dataset
        self.num_sequence = num_sequence
        self.added_time = added_time
        self.x_dx_l1,self.x_dx_l2,self.x_dx_l3 = x_dx
        self.x_rx_l1,self.x_rx_l2,self.x_rx_l3,self.x_rx_l4 = x_rx


        # formating data for model
        self.data_copy = self.df_dataset.values
        self.data_copy = self.data_copy.reshape(-1,self.num_sequence,self.df_dataset.shape[1])

        self.data_dx1 = self.x_dx_l1.values.reshape(-1,self.num_sequence,self.x_dx_l1.shape[1])
        self.data_dx2 = self.x_dx_l2.values.reshape(-1,self.num_sequence,self.x_dx_l2.shape[1])
        self.data_dx3 = self.x_dx_l3.values.reshape(-1,self.num_sequence,self.x_dx_l3.shape[1])

        self.data_rx1 = self.x_rx_l1.values.reshape(-1,self.num_sequence,self.x_rx_l1.shape[1])
        self.data_rx2 = self.x_rx_l2.values.reshape(-1,self.num_sequence,self.x_rx_l2.shape[1])
        self.data_rx3 = self.x_rx_l3.values.reshape(-1,self.num_sequence,self.x_rx_l3.shape[1])
        self.data_rx4 = self.x_rx_l4.values.reshape(-1,self.num_sequence,self.x_rx_l4.shape[1])

        self.data_final_copy = self.data_copy[:,-1,:]
        self.time = self.data_final_copy[:, 4] #tte

        self.features_demo = self.data_copy[:, :, 5:7].astype('float32')

        self.features_dx1 = self.data_dx1[:, :, 3:].astype('int16')
        self.features_dx2 = self.data_dx2[:, :, 3:].astype('int16')
        self.features_dx3 = self.data_dx3[:, :, 3:].astype('int16')

        self.features_rx1 = self.data_rx1[:, :, 3:].astype('int16')
        self.features_rx2 = self.data_rx2[:, :, 3:].astype('int16')
        self.features_rx3 = self.data_rx3[:, :, 3:].astype('int16')
        self.features_rx4 = self.data_rx4[:, :, 3:].astype('int16')

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

        x_demo = self.features_demo[index]
        x_dx1 = self.features_dx1[index]
        x_dx2 = self.features_dx2[index]
        x_dx3 = self.features_dx3[index]

        x_rx1 = self.features_rx1[index]
        x_rx2 = self.features_rx2[index]
        x_rx3 = self.features_rx3[index]
        x_rx4 = self.features_rx4[index]

        t = self.tte.reshape(-1, self.num_sequence,1)[index]
        y = self.event.reshape(-1, self.num_sequence,1)[index]
        day = self.day[index]
        m = self.mask3[index]

        return (x_dx1, x_dx2, x_dx3), (x_rx1, x_rx2, x_rx3, x_rx4), x_demo, t, y, day, m




batch_size = 2000


akidataset_train = AKIDataset(df_dataset=train_set_df_,
                              x_dx=(x_dx_l1_d_train_, x_dx_l2_d_train_, x_dx_l3_d_train_),
                              x_rx=(x_rx_l1_d_train_, x_rx_l2_d_train_, x_rx_l3_d_train_, x_rx_l4_d_train_),
                              num_sequence=5,
                              num_category=num_category,
                              num_event=num_event,
                              added_time=added_time_df_tr_['added_time'].values)



akidataset_test = AKIDataset(df_dataset=test_set_df,
                             x_dx=(x_dx_l1_d_test, x_dx_l2_d_test, x_dx_l3_d_test),
                             x_rx=(x_rx_l1_d_test, x_rx_l2_d_test, x_rx_l3_d_test, x_rx_l4_d_test),
                             num_sequence=5,
                             num_category=num_category,
                             num_event=num_event,
                             added_time=added_time_df_test['added_time'].values)


train_loader = DataLoader(dataset=akidataset_train,batch_size=batch_size,shuffle=True)
train_loader_whole = DataLoader(dataset=akidataset_train,batch_size=len(akidataset_train),shuffle=True)
test_loader_whole = DataLoader(dataset=akidataset_test,batch_size=batch_size,shuffle=True)

tr_time = np.floor(akidataset_train.tte.reshape(-1, 5,1)[:,-1,:])
tr_label = akidataset_train.label
eval_time = [int(np.percentile(tr_time, 25)), int(np.percentile(tr_time, 50)), int(np.percentile(tr_time, 75))]


##################################################################################################

#train
##############################     experiment     #############################
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

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
                embedding_demo_size=2,
                onto_dx_params=onto_dx_params,
                onto_rx_params=onto_rx_params,
                code_dim=code_dim,
                encoder_num_layers=1,
                embed_dim=embed_dim,
                rep_dim=rep_dim,
                hidden_dim=hidden_dim,
                num_event=num_event,
                num_category=num_category,
                device = device).to(device)




#optmizer
optimizer = optim.Adam(MyModel.parameters(), lr=learning_rate, weight_decay= 2 * 1e-5)
#optimizer = optim.RMSprop(MyModel.parameters(), lr=learning_rate, weight_decay= 2 * 1e-5)

# loss lists
loss_list_epoch = []
loss_list_test_epoch = []

# c-index
c_index_train_all = []
c_index_test_all = []
wighted_c_index_test_all=[]


#ranking
ranking_loss_list_epoch = []


#train
for epoch in range(num_epochs):

    loss_list = []
    loss_list_test = []

    ranking_loss_list = []

    MyModel.train()
    loop = tqdm(train_loader, total=len(train_loader), leave=False)


    pred_duration_train_list=[]
    target_label_train_list = []
    target_ett_train_list = []
    for x_dx_train,x_rx_train, x_demo_train, t_train, y_train, day_train, mask_train in loop:


        ########################### converting to pytorch tensors ###################################


        x_dx_train = [i.to(device) for i in x_dx_train]
        x_rx_train = [i.to(device) for i in x_rx_train]
        x_demo_train = x_demo_train.to(device)
        t_train = t_train.to(device)
        y_train = y_train.to(device)
        day_train = day_train.to(device)
        mask_train = mask_train.to(device)

        target_label = y_train[:,-1].reshape((-1, 1)).float()
        target_ett = t_train[:,-1].reshape((-1, 1)).float()
        target_label_train_list.append(target_label)
        target_ett_train_list.append(target_ett)




        ################################## model - train ##############################################
        optimizer.zero_grad()

        sigmoid_probs, compress_visit, att_train, e_train= MyModel(x_dx_train,x_rx_train, x_demo_train)
        surv_probs = torch.cumprod(sigmoid_probs, dim=-1)
        pred_duration_train = torch.sum(surv_probs, dim=-1)
        pred_duration_train_list.append(pred_duration_train)


        # calculating losses
        loss = OrdinalRegLoss(surv_probs, target_label, mask_train)

        ranking_loss = randomized_ranking_loss(pred_duration_train, target_label, target_ett)

        totall_loss =  loss + 120*ranking_loss
        totall_loss.backward()

        loss_list.append(loss.item())
        ranking_loss_list.append(ranking_loss.item())
        optimizer.step()



        #######################train evaluation ############################

    pred_duration_train_all = torch.cat(pred_duration_train_list, dim=0)
    target_label_train_all = torch.cat(target_label_train_list, dim=0)
    target_ett_train_all = torch.cat(target_ett_train_list, dim=0)

    c_index_train_list = []
    for i in range(num_event):
        pred_duration_train_i = pred_duration_train_all[:,i]

        ## in calculation of c-index seprately for each event
        ## if a sample is not censored and from other event than (i+1), the ett should be  (mean-life time)
        is_obsorved_train_i = (target_label_train_all == i + 1).float()
        is_obsorved_or_cencored_train_i = ((target_label_train_all == 0) | (target_label_train_all == i + 1)).float()
        n_target_ett2 = ((1-is_obsorved_or_cencored_train_i) * num_category + (is_obsorved_or_cencored_train_i * target_ett_train_all))
        n_target_ett2_1 = (n_target_ett2.squeeze()-1) * is_obsorved_train_i.squeeze() + n_target_ett2.squeeze() * (1-is_obsorved_train_i.squeeze())



        c_index_train = concordance_index(torch.floor(n_target_ett2_1).cpu().detach().numpy(),
                                          pred_duration_train_i.cpu().detach().numpy(),
                                          is_obsorved_train_i.cpu().detach().numpy())
        c_index_train_list.append(c_index_train)



    ########################################### test  #################################################
    MyModel.eval()
    with torch.no_grad():

        num_positive=0
        num_totall=0

        pred_duration_test_list=[]
        surv_probs_test_list=[]
        target_label_test_list = []
        target_ett_test_list = []
        for x_dx_test, x_rx_test, x_demo_test, t_test, y_test, day_test, mask_test in test_loader_whole:

            x_dx_test = [i.to(device) for i in x_dx_test]
            x_rx_test = [i.to(device) for i in x_rx_test]
            x_demo_test = x_demo_test.to(device)
            y_test = y_test.to(device)
            t_test = t_test.to(device)
            day_test = day_test.to(device)
            mask_test = mask_test.to(device)


            target_label_test = y_test[:,-1].reshape((-1, 1)).float()
            target_ett_test = t_test[:,-1].reshape((-1, 1)).float()
            target_day_test = day_test[:,-1].reshape((-1, 1)).float()
            target_label_test_list.append(target_label_test)
            target_ett_test_list.append(target_ett_test)

            sigmoid_probs_test, _ ,att_test, e_test= MyModel(x_dx_test, x_rx_test, x_demo_test)
            surv_probs_test = torch.cumprod(sigmoid_probs_test, dim=-1)
            pred_duration_test = torch.sum(surv_probs_test, dim=-1)

            # test loss
            loss_test = OrdinalRegLoss(surv_probs_test, target_label_test, mask_test)

            total_loss_test =  loss_test
            loss_list_test.append(loss_test.item())


            pred_duration_test_list.append(pred_duration_test)
            surv_probs_test_list.append(surv_probs_test)


        surv_probs_test_all = torch.cat(surv_probs_test_list, dim=0)
        pred_duration_test_all = torch.cat(pred_duration_test_list, dim=0)
        target_label_test_all = torch.cat(target_label_test_list, dim=0)
        target_ett_test_all = torch.cat(target_ett_test_list, dim=0)

        c_index_test_list=[]
        for i in range(num_event):
            pred_duration_test_i = pred_duration_test_all[:,i]


            ## in calculation of c-index seprately for each event
            ## if a sample is not censored and from other event than (i+1), the ett should be 212 (mean-life time)
            is_obsorved_test_i = (target_label_test_all == i + 1).float()
            is_obsorved_or_cencored_test_i = ((target_label_test_all == 0) | (target_label_test_all == i + 1)).float()
            n_last_time_test = ((1-is_obsorved_or_cencored_test_i) * 10 + (is_obsorved_or_cencored_test_i * target_ett_test_all))
            #pred_duration_test_i1 = (pred_duration_test_i+1) * is_obsorved_test_i.squeeze() + pred_duration_test_i * (1-is_obsorved_test_i.squeeze())
            n_last_time_test1 = (n_last_time_test.squeeze()-1) * is_obsorved_test_i.squeeze() + n_last_time_test.squeeze() * (1-is_obsorved_test_i.squeeze())


            c_index_test = concordance_index(torch.floor(n_last_time_test1).cpu().detach().numpy(),
                                            pred_duration_test_i.cpu().detach().numpy(),
                                            is_obsorved_test_i.cpu().detach().numpy())
            c_index_test_list.append(c_index_test)


            va_result1 = np.zeros([num_event, len(eval_time)])

            for t, t_time in enumerate(eval_time):
                eval_horizon = int(t_time)

                if eval_horizon >= num_category:
                    print('ERROR: evaluation horizon is out of range')
                else:
                    #risk = np.sum(pred[:,:,:(eval_horizon+1)], axis=2) # risk score until eval_time
                    surv_i = torch.sum(surv_probs_test_all[:,i,:(eval_horizon+1)], dim=-1)
                    for k in range(num_event):

                        va_result1[k, t] = weighted_c_index(T_train=tr_time,
                                           Y_train=tr_label.astype(int),
                                           Prediction=surv_i.cpu().detach().numpy(),
                                           T_test=torch.floor(n_last_time_test1).cpu().detach().numpy(),
                                           Y_test=is_obsorved_test_i.cpu().detach().numpy(),
                                           Time=eval_horizon)


            tmp_valid = np.mean(va_result1)



    ##############################################   print  ###############################################
    print(f"[Epoch {epoch} / {num_epochs}]")
    print(f'--train_loss = {np.mean(loss_list)} --test_loss = {np.mean(loss_list_test)}')
    print(f'--train_ranking_loss = {np.mean(ranking_loss_list)}')
    print(f'train_c-index: {c_index_train_list}\ntest_c-index: {c_index_test_list}')
    print(f'weighted_test_c-index: {tmp_valid}')

    loss_list_epoch.append(np.mean(loss_list))
    loss_list_test_epoch.append(np.mean(loss_list_test))
    ranking_loss_list_epoch.append(np.mean(ranking_loss_list))

    c_index_train_all.append(c_index_train_list)
    c_index_test_all.append(c_index_test_list)
    wighted_c_index_test_all.append(tmp_valid)




checkpoint = {
    "state_dict": MyModel.state_dict(),
}
save_checkpoint(checkpoint)