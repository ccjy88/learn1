# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:28:11 2020

@author: Administrator
600031
300142
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM

epochs=30
batch_size=32
lstm_size=128
lastdays=30
input_colnames=['high','low','close']
target_colname='target_col'
window_size=3
'''将5天close生成数组'''

#filename='300142.txt'
filename='600031.txt'


def normalise_zero_base(df):
    return df / df.iloc[0] - 1


def extract_window_data(df_train,window_size):
    x_datas=[]
    for i in range(len(df_train) - window_size):
        day5 = df_train[i:i+window_size][input_colnames];
        day5 = day5 / day5.iloc[0] - 1
        x_datas.append(day5.values)
    x_datas=np.array(x_datas)
    return x_datas


def calc_target_col(openprice,high,low,close):
    return (high - low) * 0.5 + low


f=open(filename,mode='r')

lines = f.readlines()
lines = lines[4:]

datas=[]
for line in lines:
    line = line.strip()
    words = line.split()
    if len(words)<11:
        continue
    strdate=words[0]
    strdate=strdate.replace('/','-')
    date=pd.to_datetime(strdate,format='%Y-%m-%d')
    openprice=float(words[1])
    high=float(words[2])
    low=float(words[3])
    close=float(words[4])
    target_col=calc_target_col(openprice,high,low,close)
    
    daydata=dict(date=date,
                 open=openprice,high=high,
                 low=low,close=close,
                 target_col=target_col)
    # vol=float(words[5])
    '''计算中线'''
    datas.append(daydata)

'''500天样本'''
max_days=500
df=pd.DataFrame(datas)
if len(df)>max_days :
    df = df[len(df) - max_days:]
df = df.set_index(['date'],drop=True)

#www = np.where(df.index==pd.to_datetime('2019-11-26',format='%Y-%m-%d'))
    

split_at = len(df) - lastdays
df_train=df[:split_at]
df_test=df[split_at:]




x_datas=extract_window_data(df_train,window_size)

'''将5天后close做为y'''
y_train=df_train[window_size:][target_colname].values
y_train/=df_train[:-window_size][target_colname].values
y_train-=1

'''测试数据'''
x_test_datas=extract_window_data(df_test,window_size)


np.random.seed(42)
input_dim=len(input_colnames)
model = Sequential()
model.add(LSTM(lstm_size, input_shape=(window_size,input_dim)))
model.add(Dropout(0.2))
model.add(Dense(1,activation='linear'))

model.compile(loss='mse', optimizer='adam')
model.summary()

model.fit(
    x_datas, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

pred_data = model.predict(x_test_datas)
pred_data = pred_data.squeeze()

actural_close=df_test[window_size:][target_colname]

pred_close = df_test[:-window_size][target_colname].values*(pred_data+1)
pred_close = pd.Series(pred_close,index=actural_close.index)
#line_plot(actural_close,pred_close,'actural','pred')

merge_debug=df_test[window_size:]
merge_debug['pred']=pred_close

print(merge_debug)

lw=2
fig, ax = plt.subplots(1, figsize=(15, 8))
ax.plot(merge_debug['high'], label='high', marker='o',linewidth=lw)
#ax.plot(merge_debug['close'],color='blue',marker='o', label='close', linewidth=lw)
ax.plot(pred_close,color='red',marker='o', label='pred', linewidth=lw,linestyle='dashed')
ax.plot(merge_debug['low'], label='low',marker='o', linewidth=lw)
ax.set_ylabel('price', fontsize=14)
ax.set_title(filename, fontsize=16)
ax.legend(loc='best', fontsize=16);


