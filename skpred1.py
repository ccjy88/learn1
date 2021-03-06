# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:39:40 2020

@author: wwh
"""

import sys as sys
import tracemalloc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import gc
import os
import re
import time
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM

def Readday2month(filename):
    f = open('days/'+filename+'.txt')
    lines = f.readlines()
    
    pattern = re.compile(r'^\d{4,4}/')

    records=[]
    record=dict(date=None,openprice=0,close=0,high=0,low=1e10,yearmonth='')
    for line in lines:
        if pattern.match(line)==None:
            continue
        line = line.strip()
        words = line.split()
        strdate=words[0]
        strdate=strdate.replace('/','-')
        ym=strdate[0:7]
        date=pd.to_datetime(strdate,format='%Y-%m-%d')
        openprice=float(words[1])
        high=float(words[2])
        low=float(words[3])
        close=float(words[4])

        if ym!=record['yearmonth']:
            if len(record['yearmonth'])>0:
                records.append(record)
            record=dict(date=None,openprice=0,close=0,high=0,low=1e10,yearmonth=ym)
            record['date']=date
            record['openprice']=openprice
        
        record['date']=date
        record['close']=close
        if high>record['high']:
            record['high']=high
        if low<record['low']:
            record['low']=low
     #end for
    records.append(record)
    return records

'''预测'''
class SkPred(object):
    '''用于测试最近的天数'''
    window_size=3
    lastdays=window_size+1

    epochs=100
    batch_size=32
    lstm_size=128
    input_colnames=['low']
    target_colname='low'
    max_days=500
    filename=''
    found_current=False
    
    
    def __init__(self, filename,):
        self.filename=filename
        
    def readDays(self,skcode):
        monthrec = Readday2month(skcode)
        df=pd.DataFrame(monthrec) 
        if len(df)>self.max_days :
            df = df[len(df) - self.max_days:]
        self.df = df.set_index(['date'],drop=True)

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
        
        x_datas=self.extract_window_data(df_train)
        
        '''将5天后close做为y，这里copy()的目的是为了不破坏df_train'''
        y_train=df_train[self.window_size:][self.target_colname].copy().values
        y_train/=df_train[:-self.window_size][self.target_colname].values
        y_train-=1
        
        '''测试数据'''
        x_test_datas=self.extract_window_data(df_test)
        input_dim=len(self.input_colnames)
        model = Sequential()
        model.add(LSTM(self.lstm_size, input_shape=(self.window_size,input_dim)))
        model.add(Dropout(0.2))
        model.add(Dense(1,activation='linear'))
        
        model.compile(loss='mse', optimizer='adam')
        #model.summary()
        
        model.fit(
            x_datas, y_train, epochs=self.epochs, 
            batch_size=self.batch_size, verbose=0)
        self.df_train=df_train
        self.df_test=df_test
        self.x_datas=x_datas
        self.x_test_datas=x_test_datas
        self.y_train=y_train
        self.model=model
            
            
        
    def normalise_zero_base(self,df):
        return df / df.iloc[0] - 1
    
    
    def extract_window_data(self,df_train):
        window_size=self.window_size
        x_datas=[]
        for i in range(len(df_train) - window_size):
            day5 = df_train[i:i+window_size][self.input_colnames];
            day5 = day5 / day5.iloc[0] - 1
            x_datas.append(day5.values)
        x_datas=np.array(x_datas)
        return x_datas
    
    
    def calc_target_col(self,openprice,high,low,close):
        return low
    
    def predict(self,df_test):
        model=self.model
        x_test_datas=self.extract_window_data(df_test)
        pred_data = model.predict(x_test_datas)
        pred_data = pred_data.squeeze()
        
        actural_close=df_test[self.window_size:][self.target_colname]
        
        pred_close = df_test[:-self.window_size][self.target_colname].values*(pred_data+1)
        pred_close = pd.Series(pred_close,index=actural_close.index)
        #line_plot(actural_close,pred_close,'actural','pred')
        
        merge_debug=df_test[self.window_size:].copy()
        index0=merge_debug.index[0]
        index1=merge_debug.index[len(merge_debug)-1]
 #       index1+=datetime.timedelta(days=1)
        merge_debug.loc[index0:index1,'pred']=pred_close
        
        uprates=[]
        for i in range(len(merge_debug)-1):
            i0=merge_debug.index[i]
            i1=merge_debug.index[i+1]
            c = merge_debug.loc[i0,'close']
            nexthigh=merge_debug.loc[i1,'high']
            uprates.append( (nexthigh/c-1)*100 )
        uprates.append(0)
        merge_debug.loc[index0:index1,'uprate']=np.array(uprates)

        self.merge_debug=merge_debug 
        self.pred_close=pred_close

    def drawpredict(self):
        merge_debug=self.merge_debug
        pred_close=self.pred_close
        lw=2
        fig, ax = plt.subplots(1, figsize=(15, 8))
        ax.plot(merge_debug['high'], color='gray',label='high', marker='o',linewidth=lw)
        ax.plot(merge_debug['close'],color='green',marker='o', label='close', linewidth=lw)
        ax.plot(pred_close,color='red',marker='o', label='pred', linewidth=lw,linestyle='dashed')
        #ax.plot(merge_debug['low'], label='low',marker='o', linewidth=lw)
        ax.set_ylabel('price', fontsize=14)
        ax.set_title(self.filename, fontsize=16)
        ax.legend(loc='best', fontsize=16);
        
    def trytreat(self):
        merge_debug = self.merge_debug
        predcolname='pred'
        treatresult=[]
        values=merge_debug[predcolname].values
        for i in range(1,len(merge_debug)-1):
            v0=values[i-1]
            v1=values[i]
            v2=values[i+1]
            
            if(v0>=v1 and v1<v2):
                if i==len(merge_debug)-2:
                    self.found_current=True
                onetreat=dict()
                treatresult.append(onetreat) 
                dayindex=merge_debug.index[i]
                today = merge_debug[i:i+1]
                onetreat['datetime']=dayindex
                tomorrow = merge_debug[i+1:i+2]
                th = tomorrow['high'][0]
                tc = today['close'][0]
                
                if th/tc>=1.03:
                    onetreat['percent3']=1
                else:
                    onetreat['percent3']=0
                    
                if th/tc>=1.05:
                    onetreat['percent5']=1
                else:
                    onetreat['percent5']=0
                    
                if th/tc>=1.1:
                    onetreat['percent10']=1
                else:
                    onetreat['percent10']=0
            #end of      if(v0>=v1 and v1<v2):
        #end of for
        self.treatresult=treatresult
                    
                                       
            


      
        
def doTreate(testcodes):
    for skcode in testcodes:
        print("begin read ",skcode)
        skpred=SkPred(skcode)  
        '''将最后一行数据复制，日期索引加30天'''
        skpred.readDays(skcode)
        skpred.copylastdata()
        if(len(skpred.df)<36): #数据太少
            continue
        
        skpred.train(1) #测试用最近6天，前面的用于训练
        
        '''预测'''
        df = skpred.df
        totallen=len(df)
        index0=df.index.values[totallen - 24]
        index1=df.index.values[totallen - 1]
        df_test=df.loc[index0:index1]
        
        skpred.predict(df_test)
        if len(testcodes)==1:
            '''画图表'''
            skpred.drawpredict()
            print(skpred.merge_debug)
        
        '''计算3percent 5percent'''
        skpred.trytreat()
        
        print(skcode)
        print(skpred.treatresult)
        if skpred.found_current==True:
            f=open('发现.txt','a')
            f.write(time.strftime('%Y-%m-%d %H:%M:%S '))
            f.write(skcode)
            f.write('\n')
            f.flush()
            f.close()
            print("===========!!!!! Found current {} !!!!!===========".format(skcode))
        
        del skpred,df,df_test
        del index0,index1
        gc.collect()
        
        
        
    #end if for skcode
      
    del testcodes
    gc.collect()
        

tracemalloc.start()
snapshot1 = tracemalloc.take_snapshot()

dirname='days'        
testcodes=[]
pattern = re.compile(r'\d{6,6}.txt')
maxcount=25
skipcode=''

skcodeset=set()
for filename in os.listdir(dirname):
    if pattern.match(filename):
        skcode=filename[0:6]
        if skcode not in skcodeset:
            skcodeset.add(skcode)
skcodes=sorted(skcodeset)


for skcode in skcodes:
    if (skcode <= skipcode):
        continue
    if(maxcount==0 ):
        break
    maxcount-=1
    testcodes.append(skcode)
        
   
'''手工指定'''
testcodes=['000858']     

doTreate(testcodes)        

snapshot2 = tracemalloc.take_snapshot()
top_stats = snapshot2.compare_to(snapshot1, 'lineno')
#print(top_stats)
    
print("finished,exit...")

