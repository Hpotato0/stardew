import warnings
import os, random
import tensorflow as tf
import numpy as np
import pandas as pd

from dataset import Dataset, TrainDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from trainer import Trainer
from utility import astype_data
from model import Transformer
from lists import data_list, tr_del_list, data_list_without_imexport

# 경고 끄기
warnings.filterwarnings(action='ignore')

# 시드고정
tf.random.set_seed(19970119)
random.seed(19970119)
np.random.seed(19970119)

# Preprocessing; 처음에 한번만 하면 됨
# dataset = Dataset()

# epoch = 1000
# batch = 15
epoch = 10
batch = 128


## Train 과정
# General model (with imexport) 만들고
xdatas = []
ydatas = []

for i in tqdm(data_list):
    traindata = TrainDataset(tr_del_list, i)
    if len(xdatas) == 0:
        xdatas = traindata.xdata
        ydatas = traindata.ydata
    else:
        xdatas = np.concatenate((xdatas, traindata.xdata), axis=0)
        ydatas = np.concatenate((ydatas, traindata.ydata), axis=0)

# train, validation 분리 (8 : 2)
x_train, x_val, y_train, y_val = train_test_split(xdatas, ydatas, test_size=0.5, shuffle=True, random_state=42)

del xdatas
del ydatas

model = Transformer(x_train, 'general', epoch, batch)
trainer = Trainer(model, astype_data(x_train), astype_data(y_train), astype_data(x_val), astype_data(y_val),
                    batch, name=f'transformer-general')

if not model.loaded:
# transformer 모델 훈련 -> 왜 각 농산물마다 다른 모델을 쓸까?
    trainer.train(epoch)
model.save_model('general', epoch, batch)

# General model (wihtout imexport) 만들고
xdatas = []
ydatas = []

for i in tqdm(data_list_without_imexport):
    traindata = TrainDataset(tr_del_list, i)
    if len(xdatas) == 0:
        xdatas = traindata.xdata
        ydatas = traindata.ydata
    else:
        xdatas = np.concatenate((xdatas, traindata.xdata), axis=0)
        ydatas = np.concatenate((ydatas, traindata.ydata), axis=0)

# train, validation 분리 (8 : 2)
x_train, x_val, y_train, y_val = train_test_split(xdatas, ydatas, test_size=0.5, shuffle=True, random_state=42)

del xdatas
del ydatas

model = Transformer(x_train, 'general-without', epoch, batch)
trainer = Trainer(model, astype_data(x_train), astype_data(y_train), astype_data(x_val), astype_data(y_val),
                    batch, name=f'transformer-general-without')

if not model.loaded:
# transformer 모델 훈련 -> 왜 각 농산물마다 다른 모델을 쓸까?
    trainer.train(epoch)
model.save_model('general-without', epoch, batch)
yy_val = trainer.model.predict(x_val)
print(x_val[0][0], x_val[1][0], y_val[0], y_val[1], yy_val[0], yy_val[1])

# Finetuning
for i in tqdm(data_list):
    print(i)
    traindata = TrainDataset(tr_del_list, i)
    df_number = traindata.df_number

    # train, validation 분리 (8 : 2)
    x_train, x_val, y_train, y_val = train_test_split(traindata.xdata, traindata.ydata, test_size=0.2, shuffle=True, random_state=42)

    model = Transformer(x_train, df_number, epoch, batch)
    trainer = Trainer(model, astype_data(x_train), astype_data(y_train), astype_data(x_val), astype_data(y_val),
                        batch, name=f'transformer-{df_number}')
    
    if not model.loaded:
    # transformer 모델 훈련 -> 왜 각 농산물마다 다른 모델을 쓸까?
        trainer.train(epoch)
    model.save_model(df_number, epoch, batch)

for i in tqdm(data_list_without_imexport):
    print(i)
    traindata = TrainDataset(tr_del_list, i)
    df_number = traindata.df_number

    # train, validation 분리 (8 : 2)
    x_train, x_val, y_train, y_val = train_test_split(traindata.xdata, traindata.ydata, test_size=0.2, shuffle=True, random_state=42)

    model = Transformer(x_train, df_number, epoch, batch)
    trainer = Trainer(model, astype_data(x_train), astype_data(y_train), astype_data(x_val), astype_data(y_val),
                        batch, name=f'transformer-{df_number}')

    if not model.loaded:
    # transformer 모델 훈련 -> 왜 각 농산물마다 다른 모델을 쓸까?
        trainer.train(epoch)
    model.save_model(df_number, epoch, batch)
    
