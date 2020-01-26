# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 10:47:30 2020

@author: wwh
读入文件格式
skcode,yearmm[-endmonth],label
"""


import re
import math
import pandas as pd
from pandas import DataFrame
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


def Checkdefine(df):
    checklist=dict()
    checklist['ding']=set(['down','plat','up'])
    checklist['up']=set(['up','up1','ding','plat'])
    checklist['up1']=set(['up','plat','ding'])
    checklist['down']=set(['plat','up','down','up1'])
    checklist['down1']=set(['up','down','plat'])
    checklist['plat']=set(['up1','down1','plat'])
    
    
    index=df.index
    for i in range(len(index)-1):
        skcode=df.loc[index[i],'skcode']
        next_skcode=df.loc[index[i+1],'skcode']
        
        if skcode != next_skcode :
            continue
        
        year=df.loc[index[i],'year']
        month=df.loc[index[i],'month']
        label=df.loc[index[i],'label']
        next_label=df.loc[index[i+1],'label']
        labelset = checklist[label]
        if next_label not in labelset:
            print('skcode={} yearmonth={}-{},label={} next cannot  {}'.format(skcode,year,month,label,next_label))
        

def ReadlabelDefine():
    f = open('sklabel1.txt',mode='r')
    pattern = re.compile(r'^\d{6,6},')
    records=[]
    for line in f.readlines():
        line=line.strip()
        if pattern.match(line)==None:
            continue
        words = line.split(',')
        if(len(words)!=3):
            raise('error line='+line)
        skcode=words[0]
        period=words[1]
        label=words[2]
        '''分解日期'''
        periods=period.split('-')
        if len(periods)==1:
            rec=dict(skcode=skcode,year=int(period[0:4]),month=int(period[4:6]),label=label)
            records.append(rec)
        else:
            m1=int(periods[0][4:6])
            m2=int(periods[1])
            
            for m in range(m1,m2+1):
                rec=dict(skcode=skcode,year=int(period[0:4]),month=m,label=label)
                records.append(rec)
    f.close()
    df=DataFrame(records)
    Checkdefine(df)
    return df            
        
        

def Readday2month(skcode):
    f = open('days/'+skcode+'.txt')
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
    

    f.close()
    return records

def calcSklog10(df,skcode,yearmonth,windowsize):
    fd=False
    for i in df.index:
        rec=df.loc[i]
        if(rec['yearmonth'] == yearmonth):
            fd=True
            break;
    if(fd):
        dfsub=df.loc[i-windowsize+1:i]
    else:
        print("cannot found {} {}".format(skcode,yearmonth))
        return []
    
    datacolnames=['close','high','low','openprice']
    closes = dfsub.loc[:,datacolnames].values    
    closes = np.array(closes).reshape((closes.shape[0] * closes.shape[1],))
    log10v =[math.log10(x) for x in closes]
    log10v = np.array(log10v)
    minv = log10v.min()
    log10v = log10v - minv
    return log10v

df_define = ReadlabelDefine()
windowsize=6


mem_skcode=''
train_datas=[]
label_datas=[]
for row in df_define.index:
    skcode=df_define.loc[row,'skcode']
    y=df_define.loc[row,'year']
    m=df_define.loc[row,'month']
    yearmonth = '%04d-%02d' % (y,m)
    label=df_define.loc[row,'label']
    if(mem_skcode != skcode):
        df_data = DataFrame(Readday2month(skcode))
        mem_skcode = skcode
    log10data=calcSklog10(df_data,skcode,yearmonth,windowsize)
    if len(log10data) > 0 :
        train_datas.append(log10data)
        label_datas.append(label)
    
x_train=np.array(train_datas)
encoder=LabelEncoder()
encoder.fit(label_datas)
encoded_Y=encoder.transform(label_datas)
maxclasses = encoded_Y.max()+1
y_train=np_utils.to_categorical(encoded_Y)



model = Sequential()
model.add(Dense(256,activation='relu',input_shape=(x_train.shape[1],)))
model.add(Dense(128,activation='relu'))
model.add(Dense(maxclasses,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=150,verbose=1,batch_size=1)

testdatas=[]
skcode='002019'
df_data = DataFrame(Readday2month(skcode))

#testdatas.append(calcSklog10(df_data,skcode,'2020-01',windowsize))

test_y=2019
out_ym=[]
for test_m in range(1,13):
    yearmonth="%4d-%02d" %(test_y,test_m)
    out_ym.append(yearmonth)
    testdatas.append(calcSklog10(df_data,skcode,yearmonth,windowsize))


testdatas.append(calcSklog10(df_data,skcode,'2020-01',windowsize))
out_ym.append('2020-01')

x_test=np.array(testdatas)

preds = model.predict(x_test)
preds_class = model.predict_classes(x_test)

pred_label = encoder.inverse_transform(preds_class)

for i,label in enumerate(pred_label):
    print('{} {}'.format(out_ym[i],label))
