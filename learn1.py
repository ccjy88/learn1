# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:07:52 2020

@author: Administrator
"""
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras import layers


digitstr = '0123456789+ '
sortlist=sorted(list(digitstr))
char_index = dict( (c,i) for i,c in enumerate (sortlist) )
index_char = dict( (i,c) for i,c in enumerate (sortlist) )


DATA_DIMS=len(digitstr)
#2位数
DIGITS=2
INPUT_STEPS=DIGITS+1+DIGITS #输入最长5步。
OUTPUT_STEPS=3  #输出最大为3位数。

#生成BOOL型1维数组。
def encode(c):
    x = np.zeros( (DATA_DIMS),dtype=np.bool)
    x[char_index.get(c)]=True
    return x;

def decode(indexes):
    s = ''.join(index_char[i] for i in indexes)
    return s
        


def create_hot_one(inputstring,n_rows,n_cols):
    x = np.zeros( (n_rows,n_cols),dtype=np.bool)
    for i,c in enumerate(inputstring):
        x[i]=encode(c)
    return x

def test_add(a,b):
    sentence=str(a)+'+'+str(b)
    sentence+=' '*(INPUT_STEPS - len(sentence))
    sentence=sentence[::-1]
    x = np.zeros((1,INPUT_STEPS,DATA_DIMS),dtype=np.bool)
    x[0]=create_hot_one(sentence,INPUT_STEPS,DATA_DIMS)
    y_predict =  model.predict_classes(x,batch_size=BATCH_SIZE)
    guess = decode(y_predict[0])
    return guess


questions=[]
answers=[]


i_nums=np.linspace(1,99,num=99,dtype=np.integer)
np.random.shuffle(i_nums)

j_nums=np.linspace(1,99,num=99,dtype=np.integer)
np.random.shuffle(j_nums)

for i in i_nums:
    for j in j_nums:
        sentence=str(i)+'+'+str(j)
        
        #补足长度
        sentence+=' '*(INPUT_STEPS - len(sentence))
        #倒转
        sentence=sentence[::-1]
        questions.append(sentence)
        ans=str(i+j)
        ans+=' '*(OUTPUT_STEPS - len(ans))
        answers.append(ans)

sample_count=len(questions)

x_samples=np.zeros((sample_count,INPUT_STEPS,DATA_DIMS),dtype=np.bool)
y_samples=np.zeros((sample_count,OUTPUT_STEPS,DATA_DIMS),dtype=np.bool)

#生成BOOL型 shape(采样数量 ， 输入步数 ， 每一步数据维度)
for  i,q in enumerate (questions) :
   x_samples[i] = create_hot_one(q , INPUT_STEPS,DATA_DIMS)
   y_samples[i] = create_hot_one( answers[i],OUTPUT_STEPS,DATA_DIMS  )

split_at = sample_count - sample_count//10
x_tran,y_tran=x_samples[:split_at],y_samples[:split_at]
x_test,y_test=x_samples[split_at:],y_samples[split_at:]
answer_test=answers[split_at:]

#开始建立MODEL
HIDDEN_SIZE=128
BATCH_SIZE=128

model = Sequential()
model.add(LSTM(HIDDEN_SIZE,input_shape=(INPUT_STEPS,DATA_DIMS)))
model.add(RepeatVector(OUTPUT_STEPS))
model.add(LSTM(HIDDEN_SIZE,return_sequences=True))
model.add(TimeDistributed(Dense(DATA_DIMS,activation='softmax')))

model.compile(loss='categorical_crossentropy',optimizer='adam',
             metrics=['accuracy'] )
model.summary()

#重复计算多少次？
EPOCHS = 1

TRANCOUNT=80
for t in np.arange(TRANCOUNT):
    model.fit(x_tran,y_tran,batch_size=BATCH_SIZE,epochs=EPOCHS)

#test_add(1,2)

#测试
testcount = len(x_test)
correctcount=0
for i in np.arange(testcount):
    predict_x = x_test[np.array([i])]
    predict_y = model.predict_classes(predict_x,batch_size=BATCH_SIZE)
    
    correct = answer_test[i]
    guess = decode(predict_y[0])
    
    if(correct==guess):
        correctcount+=1

print("total test",testcount,"correctcount=",correctcount,"percent",(100*correctcount/testcount),'%')


        