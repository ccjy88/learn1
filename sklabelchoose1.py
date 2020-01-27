# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:12:11 2020

@author: benb wwh

读取保存的model文件，对8个周期内查找up3 无down 
"""
import re
import math
import pandas as pd
from pandas import DataFrame
from pandas import Series
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import os
from pandas import concat
import time



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

 
model = load_model('sklabel1.h5')
encoder=LabelEncoder()
encoder.fit(['ding', 'down' ,'down1', 'plat', 'up', 'up1', 'up3'])
windowsize=6



#testdatas.append(calcSklog10(df_data,skcode,'2020-01',windowsize))

def Choose(skcode,yearmonth,monthcount):
    testdatas=[]
    df_data = DataFrame(Readday2month(skcode))
    ymindex=-1
    for i,index in enumerate(df_data.index):
        if df_data.loc[index,'yearmonth']==yearmonth:
            ymindex=i
            break
    if ymindex<0:
        print("not found skcode={},yearmonth={}".format(skcode,yearmonth))
        return
    
    out_ym=[]
    for i in range(ymindex-monthcount+1,ymindex+1):
        index=df_data.index[i]
        tmpym=df_data.loc[index,'yearmonth']
        out_ym.append(tmpym)                
        testdatas.append(calcSklog10(df_data,skcode,tmpym,windowsize))
    
    
    x_test=np.array(testdatas)
#    preds = model.predict(x_test)
    preds_class = model.predict_classes(x_test)
    pred_labels = encoder.inverse_transform(preds_class)
    cols=list()
    cols.append(Series(out_ym))
    cols.append(Series(pred_labels))
    df_label = concat(cols,axis=1)
    
    colnames=list()
    colnames.append('yearmonth')
    colnames.append('label')
    df_label.columns=colnames
    return df_label
    

dirname='days'        
testcodes=[]
pattern = re.compile(r'\d{6,6}.txt')
maxcount=300
skipcode='000333'


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
# testcodes=['600745']
nowyearmonth = '2020-01'

for skcode in testcodes:
   print('parsing {}'.format(skcode))
   df_label= Choose(skcode,nowyearmonth,8)
   if(len(testcodes)==1):
       print(df_label)
   
   for index in df_label.index:
       label = df_label.loc[index,'label']
       if label=="up3":
           #print("skcode {} has up3,now is {}".format(skcode,yearmonthlabel.get('2020-01')))
           #最近1或2个是plat
           last1yearmonth=df_label.loc[df_label.index[-1],'yearmonth']
           last1label=df_label.loc[df_label.index[-1],'label']
           last2label=df_label.loc[df_label.index[-2],'label']
           if last1label=='plat' or last1label=='down' or last1label=='down1' or last2label=='plat' or last2label=='down' or last2label=='down1':
               print("!!!!!!!!!!!!!!! {} has up3".format(skcode))
               f=open('up3形态.txt','a')
               f.write(time.strftime('%Y-%m-%d %H:%M:%S '))
               f.write('{} stockdate={}\n'.format(skcode,last1yearmonth))
               f.flush()
               f.close()
               break #end of df_label
               
           
           
