# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:39:40 2020

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM

'''预测'''
class SkPred(object):
    '''用于测试最近的天数'''
    lastdays=12

    epochs=100
    batch_size=32
    lstm_size=128
    input_colnames=['low']
    target_colname='target_col'
    window_size=3
    max_days=500
    filename=''
    def __init__(self, filename,):
        self.filename=filename
        self.readfile()
        
    def readfile(self):
        f=open(self.filename,mode='r')
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
            target_col=self.calc_target_col(openprice,high,low,close)
            
            daydata=dict(date=date,
                         open=openprice,high=high,
                         low=low,close=close,
                         target_col=target_col)
            # vol=float(words[5])
            '''计算中线'''
            datas.append(daydata)
            df=pd.DataFrame(datas) 
            if len(df)>self.max_days :
                df = df[len(df) - self.max_days:]
            self.df = df.set_index(['date'],drop=True)

    '''复制最后1行'''
    def copylastdata(self):
        df=self.df
        lastrec = df.iloc[len(df) - 1].copy()
        dayindex=df.index[len(df) - 1]
        dayindex+=datetime.timedelta(days=30)
        df.loc[dayindex]=lastrec
        
        


    def train(self,lastdays=12):
        self.lastdays=lastdays
        df=self.df
        split_at = len(df) - lastdays
        df_train=df[:split_at]
        df_test=df[split_at:]
        
        x_datas=self.extract_window_data(df_train,self.window_size)
        
        '''将5天后close做为y'''
        y_train=df_train[self.window_size:][self.target_colname].values
        y_train/=df_train[:-self.window_size][self.target_colname].values
        y_train-=1
        
        '''测试数据'''
        x_test_datas=self.extract_window_data(df_test,self.window_size)
        input_dim=len(self.input_colnames)
        model = Sequential()
        model.add(LSTM(self.lstm_size, input_shape=(self.window_size,input_dim)))
        model.add(Dropout(0.2))
        model.add(Dense(1,activation='linear'))
        
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        
        model.fit(
            x_datas, y_train, epochs=self.epochs, 
            batch_size=self.batch_size, verbose=1)
        self.df_train=df_train
        self.df_test=df_test
        self.x_datas=x_datas
        self.x_test_datas=x_test_datas
        self.y_train=y_train
        self.model=model
            
            
        
    def normalise_zero_base(self,df):
        return df / df.iloc[0] - 1
    
    
    def extract_window_data(self,df_train,window_size):
        x_datas=[]
        for i in range(len(df_train) - window_size):
            day5 = df_train[i:i+window_size][self.input_colnames];
            day5 = day5 / day5.iloc[0] - 1
            x_datas.append(day5.values)
        x_datas=np.array(x_datas)
        return x_datas
    
    
    def calc_target_col(self,openprice,high,low,close):
        return low
    
    def predict(self):
        model=self.model
        pred_data = model.predict(self.x_test_datas)
        pred_data = pred_data.squeeze()
        
        df_test=self.df_test
        actural_close=df_test[self.window_size:][self.target_colname]
        
        pred_close = df_test[:-self.window_size][self.target_colname].values*(pred_data+1)
        pred_close = pd.Series(pred_close,index=actural_close.index)
        #line_plot(actural_close,pred_close,'actural','pred')
        
        merge_debug=df_test[self.window_size:]
        merge_debug['pred']=pred_close
        
        merge_debug['uprate']=np.zeros((len(merge_debug),),dtype='float64')
        for i in range(len(merge_debug)-1):
            c = merge_debug['close'][i]
            nexthigh=merge_debug['high'][i+1]
            merge_debug['uprate'][i]=(nexthigh/c-1) * 100
        print(merge_debug)
        self.merge_debug=merge_debug 
        self.pred_close=pred_close

    def drawpredict(self):
        merge_debug=self.merge_debug
        pred_close=self.pred_close
        lw=2
        fig, ax = plt.subplots(1, figsize=(15, 8))
        ax.plot(merge_debug['high'], color='yellow',label='high', marker='o',linewidth=lw)
        ax.plot(merge_debug['close'],color='green',marker='o', label='close', linewidth=lw)
        ax.plot(pred_close,color='red',marker='o', label='pred', linewidth=lw,linestyle='dashed')
        #ax.plot(merge_debug['low'], label='low',marker='o', linewidth=lw)
        ax.set_ylabel('price', fontsize=14)
        ax.set_title(self.filename, fontsize=16)
        ax.legend(loc='best', fontsize=16);
        
        
skpred=SkPred('600031m.txt')  
'''将最后一行数据复制，日期索引加30天'''
skpred.copylastdata()
skpred.train(12) #12天
skpred.predict()
skpred.drawpredict()
'''预测'''




        
        
        