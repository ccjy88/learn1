# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:48:50 2020

预测sin函数
@author: Administrator
"""


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM



def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price [CAD]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16);
def normalise_zero_base(df):
    return df / df.iloc[0] - 1


def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

x=np.arange(0,  4 * math.pi,0.01)
y=np.zeros((len(x),),dtype='float64')

for i in range(0,len(x)):
    y[i]=math.sin(x[i]) + 0.01
    
sinx = pd.DataFrame()
sinx['sinx']=y

'''90%训练'''
split_at = len(sinx) - len(sinx)//10
x_train=sinx[:split_at]
x_test=sinx[split_at:]

line_plot(x_train,x_test,'train','test')

window_len=5

x_train_array = extract_window_data(x_train,window_len,zero_base=False)
x_test_array = extract_window_data(x_test,window_len,zero_base=False)

y_train_array = x_train['sinx'][window_len:].values
y_train_array /= x_train['sinx'][:-window_len].values 
y_train_array -= 1




model = Sequential()
model.add(LSTM(128, input_shape=(window_len,1)))
#model.add(Dropout(0.1))
model.add(Dense(1,activation='linear'))

model.compile(loss='mse', optimizer='adam')
model.summary()

epochs=10
batch_size=32
model.fit(
    x_train_array, y_train_array, epochs=epochs, batch_size=batch_size, verbose=1)

'''预测'''
targets = x_test['sinx'][window_len:]
preds = model.predict(x_test_array).squeeze()
preds = x_test['sinx'].values[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)

#line_plot(targets, preds, 'actual', 'prediction', lw=3)

target_merge = pd.DataFrame(data=targets,index=targets.index)
target_merge['pred']=preds
target_merge['diff']=(target_merge['pred'].values / target_merge['sinx'].values-1) * 100
#print(target_merge)

