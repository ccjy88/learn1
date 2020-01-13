import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import numpy as np


'''
https://min-api.cryptocompare.com/data/histoday?fsym=BTC&tsym=CAD&limit=500
返回
"Response":"Success","Type":100,"Aggregated":false,"Data":[
{"time":1535673600,"close":9554.65,"high":9871.08,"low":9355.69,"open":9421.07,"volumefrom":224.36,"volumeto":2123919.83},
{"time":1535760000,"close":9722.91,"high":10010.71,"low":9349.32,"open":9542.18,"volumefrom":148.12,"volumeto":1421194.86},
{"time":1535846400,"close":9929.94,"high":9982.58,"low":9566.19,"open":9738.44,"volumefrom":213.59,"volumeto":2085236.01}
.....
'''

limit_count=800
endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=BTC&tsym=CAD&limit='+str(limit_count))
content = json.loads(res.content)['Data']
hist = pd.DataFrame(content)
###print(hist)
### hist shape=(501,7)
###columnis: close high low  open time valuemefrom volumnto

#将time做为索引列，转为datetime类型
#print(hist.index) #rangindex 0,500
hist = hist.set_index('time')
#print(hist.index) #int64index([1535673600,.....])
hist.index = pd.to_datetime(hist.index, unit='s')
#print(hist.index) # DatetimeIndex(['2018-08-31', '2018-09-01', '2018-09-02'

target_col = 'close'

hist.head(5)

#80%训练， 20%用于测试
def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

train, test = train_test_split(hist, test_size=0.2)

#绘图
def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price [CAD]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16);



line_plot(train[target_col], test[target_col], 'training', 'test', title='')

'''取第0个元素归1'''
def normalise_zero_base(df):
    return df / df.iloc[0] - 1

'''取最大最小值之间的区间归1'''
def normalise_min_max(df):
    return (df - df.min()) / (data.max() - df.min())


'''取窗口数据，从0到len(df)-窗口大小，取当前位置到加窗口大小的数组，再按zero归1'''
def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)



def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    '''取80%的训炼数据'''
    train_data, test_data = train_test_split(df, test_size=test_size)
    
    '''训炼数据按窗口生成X_train数组'''
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    
    '''取close列的值做为y'''
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values
    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1

    return train_data, test_data, X_train, X_test, y_train, y_test


def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear',
                     dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    model.summary()
    return model


np.random.seed(42)
window_len = 5
test_size = 0.2
zero_base = True
#lstm_neurons = 100
lstm_neurons = 100
#epochs = 20
epochs = 20
batch_size = 32
loss = 'mse'
dropout = 0.2
optimizer = 'adam'

train, test, X_train, X_test, y_train, y_test = prepare_data(
    hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)


model = build_lstm_model(
    X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
    optimizer=optimizer)

model = Sequential()
model.add(LSTM(128, input_shape=(window_len,6)))
#model.add(Dropout(0.2))
model.add(Dense(1,activation='linear'))

model.compile(loss='mse', optimizer='adam')

history = model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)


targets = test[target_col][window_len:]
preds_tmp = model.predict(X_test)
preds = model.predict(X_test).squeeze()
mean_absolute_error(preds, y_test)

preds = test[target_col].values[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)

target_merge = pd.DataFrame(data=targets,index=targets.index)
target_merge['pred']=preds
target_merge['diff']=(target_merge['pred'].values / target_merge['close'].values-1) * 100


print(target_merge)

line_plot(targets, preds, 'actual', 'prediction', lw=3)
