# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:51:32 2020

@author: Administrator
"""

import sys as sys
import tracemalloc
import pandas as pd
from pandas import concat
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import datetime
import gc
import os
import re
import time
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out + 1):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def Readday2month(filename):
    f = open('days/'+filename+'.txt')
    lines = f.readlines()
    
    pattern = re.compile(r'^\d{4,4}/')

    records=[]
    record=dict(date=None,openprice=0,close=0,high=0,low=1e10,yearmonth='')
    for line in lines:
        line=line.strip()
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
            record=dict(date=None,openprice=0,close=0,high=0,low=1e10,
                        yearmonth=ym,pred_low=0)
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
    '''用于测试的列数，最小3'''
    window_size=3
    lastdays=window_size+1

    epochs=100
    batch_size=10
    lstm_size=128
    target_colname=['close']
    filename=''
    
    
    def __init__(self, filename,):
        self.filename=filename
        
    def readDay2month(self,skcode):
        monthrec = Readday2month(skcode)
        df=pd.DataFrame(monthrec) 
        
        '''复制最后1行'''
        lastrec = df.iloc[len(df) - 1].copy()
        date=lastrec['date']
        date += datetime.timedelta(days=30)
        strdate = date.strftime('%Y-%m')
        lastrec['date']=date
        lastrec['yearmonth']=strdate
        newindex=len(df)
        df.loc[newindex]=lastrec
        
        '''归1'''
        supervised_data = series_to_supervised(df.loc[:,self.target_colname].values, 
            self.window_size-1, 1 ,dropnan=True)

        self.df=df[self.window_size:]
        
        calcols=[]
        
        for var in range(1,len(self.target_colname)+1):
            for t in range(-2,1,1):
                if t == 0 :
                    calcols.append('var{}(t)'.format(var))
                else:
                    calcols.append('var{}(t{})'.format(var,t))
        
        calcols=np.array(calcols)
        
        low_x = supervised_data.loc[:,calcols]
        low_x = np.array(low_x.values)
        scaler=MinMaxScaler()
        low_x = scaler.fit_transform(low_x)
        
        
        low_y = supervised_data.loc[:,['var1(t+1)']]
        low_y = np.array(low_y.values)

        self.y_scaler=MinMaxScaler()
        low_y = self.y_scaler.fit_transform(low_y)

        #abcdef=self.y_scaler.inverse_transform(low_y)


        self.low_x = low_x.reshape(len(low_x),1,low_x.shape[1])
        del low_x
        self.low_y = low_y
        del low_y
        
        del df,supervised_data
        
        
        
        

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

            '''target_col=self.calc_target_col(openprice,high,low,close)
            daydata=dict(date=date,
                         open=openprice,high=high,
                         low=low,close=close,
                         target_col=target_col)'''
            
            # vol=float(words[5])
            daydata=dict(date=date,low=low,close=close)
            datas.append(daydata)

        df=pd.DataFrame(datas) 
        self.df = df.set_index(['date'],drop=True)

        
        


    def train(self,lastdays=12):
        self.lastdays=lastdays
        splitat=len(self.low_x) - lastdays
        x_train,y_train=self.low_x[:-2],self.low_y[:-2]
        x_test,y_test=self.low_x[splitat:],self.low_y[splitat:]
        
        time_steps=x_train.shape[1]
        input_dim=x_train.shape[2]
        
        model = Sequential()
        model.add(LSTM(self.lstm_size, activation='relu',input_shape=(time_steps,input_dim)))
        model.add(Dropout(0.2))
        model.add(Dense(1,activation='linear'))
        
        model.compile(loss='mean_squared_error', optimizer='adam')
        #model.summary()
        #validation_data=(x_test,y_test)
        model.fit(
            x_train, y_train, epochs=self.epochs, 
            batch_size=self.batch_size, verbose=0)
        
        loss = model.evaluate(x_test,y_test, verbose=0)
        print('loss={}'.format(loss))
        
        self.model=model
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        
        del x_train,y_train,x_test,y_test
        
        self.df_pred=DataFrame(self.df[len(self.df)-lastdays:])
        
    
    def predict(self):
        pred_data = self.model.predict(self.x_test)
#        pred_data = pred_data.squeeze()
        pred_low = self.y_scaler.inverse_transform(pred_data)
       # index0=self.df_pred.index[0]
       # index1=self.df_pred.index[len(self.df_pred)-1]
        self.df_pred.loc[:,'pred_low']=pred_low
        
        uprates=[]
        for i in range(len(self.df_pred)-1):
            i0=self.df_pred.index[i]
            i1=self.df_pred.index[i+1]
            c = self.df_pred.loc[i0,'close']
            nexthigh=self.df_pred.loc[i1,'high']
            uprates.append( (nexthigh/c-1)*100 )
        uprates.append(0)
        index0=self.df_pred.index[0]
        index1=self.df_pred.index[len(self.df_pred)-1]
        self.df_pred.loc[index0:index1,'uprate']=np.array(uprates)
        self.df_pred = self.df_pred.set_index(['date'],drop=True)


    def drawpredict(self):
        lw=2
       # plt.figure()
        fig, ax = plt.subplots(1, figsize=(15, 8))
        ax.plot(self.df_pred['high'], color='gray',label='high', marker='o',linewidth=lw)
        ax.plot(self.df_pred['close'],color='green',marker='o', label='close', linewidth=lw)
        ax.plot(self.df_pred['pred_low'],color='red',marker='o', label='pred', linewidth=lw,linestyle='dashed')
#        ax.plot(self.df_pred['low'],color='yellow',marker='o', label='low', linewidth=lw)
        ax.set_ylabel('price', fontsize=14)
        ax.set_title(self.filename, fontsize=16)
#       ax.legend(loc='best', fontsize=16);
        
    def choose(self,skcode):
        predcolname='pred_low'
        values=self.df_pred[predcolname].values
        '''3个月新高'''
        if values[-1]>np.array(values[-4:-1]).max() and (values[-2]<=values[-3] or values[-3]<=values[-4]):
            print(skcode)
            f=open('发现.txt','a')
            f.write(time.strftime('%Y-%m-%d %H:%M:%S '))
            f.write(skcode)
            f.write('\n')
            f.flush()
            f.close()
            print("===========!!!!! Found current {} !!!!!===========".format(skcode))
                    

      
        
def doTreate(testcodes):
    for skcode in testcodes:
        print("begin read ",skcode)
        skpred=SkPred(skcode)  
        '''将最后一行数据复制，日期索引加30天'''
        skpred.readDay2month(skcode)
        if(len(skpred.df)<24): #数据太少
            continue

        if len(testcodes)>1:
            skpred.train(8) #批量固定8
        else:
            skpred.train(18) #
        '''预测'''
        skpred.predict()

        if len(testcodes)==1:
            '''画图表'''
            skpred.drawpredict()
            print(skpred.df_pred)
        
        '''计算3percent 5percent'''
        skpred.choose(skcode)
        
        del skpred
        gc.collect()
        
        
    #end if for skcode
      
    del testcodes
    gc.collect()
        

tracemalloc.start()
snapshot1 = tracemalloc.take_snapshot()

dirname='days'        
testcodes=[]
pattern = re.compile(r'\d{6,6}.txt')
maxcount=30
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
testcodes=['002019']
doTreate(testcodes)        

snapshot2 = tracemalloc.take_snapshot()
top_stats = snapshot2.compare_to(snapshot1, 'lineno')
#print(top_stats)
    
print("finished,exit...")

