# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt # Some useful functions
from torch.utils.data import DataLoader, Dataset
from pycox.datasets import metabric
from pycox.models import LogisticHazard
from pycox.models import PMF
from pycox.models import MTLR
import pycox.models as models
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
from utils_attack import *
from utils_SA import *


#load dataset
#importing AKI dataset

path = '/data_samples/'
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



# some proprocessing
np.random.seed(1234)
_ = torch.manual_seed(123)

x_train = akidataset_train.features
x_train = x_train.reshape(x_train.shape[0], -1)
y_train = akidataset_train.event[:, -1]
t_train = akidataset_train.tte[:, -1]


x_test = akidataset_test.features
x_test = x_test.reshape(x_test.shape[0], -1)
y_test = akidataset_test.event[:, -1]
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


#train model
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
batch_size = 512
lr_finder = model_DeepHitSingle.lr_finder(x_train, y_tr, batch_size, tolerance=3)
_ = lr_finder.plot()

model_DeepHitSingle.optimizer.set_lr(0.01)

epochs = 200
callbacks = [tt.cb.EarlyStopping()]
log = model_DeepHitSingle.fit(x_train, y_tr, batch_size, epochs, callbacks, val_data=test)

#evaluation -> c-index
surv = model_DeepHitSingle.predict_surv_df(x_test)
ev = EvalSurv(surv, t_test, y_test, censor_surv='km')
print(f'pycox c-index: {ev.concordance_td("antolini")}')

model_DeepHitSingle.save_model_weights("checkpoints/checkpoint_deephit.pth.tar")