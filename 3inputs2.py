# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:52:34 2017

@author: ruili2
"""
from __future__ import division
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input, concatenate, Reshape, RepeatVector, TimeDistributed, Bidirectional, Flatten, Dropout, multiply, add
from keras.layers import LSTM, dot
from keras import layers
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.models import Model
# fix random seed for reproducibility
import pickle

with open('./24012018/user_tplist.pkl', 'rb') as fid:
    tplist = pickle.load(fid)
with open('./24012018/user_tptimelist.pkl', 'rb') as fid:
    timelist = pickle.load(fid)
with open('./24012018/user_tpplist.pkl', 'rb') as fid:
    tpplist = pickle.load(fid)
with open('./24012018/label.pkl', 'rb') as fid:
    y = pickle.load(fid)
  
tplist = [ [j+1 for j in i] for i in tplist]

maxnumber_of_tp = max([max(i) for i in tplist]) + 1
print('max tp value:',maxnumber_of_tp)
max_tps_length = max([len(i) for i in tplist])
print('max tp length:',max_tps_length)
maxvalue_of_time = max([max(i) for i in timelist]) + 1
print('max time value:',maxvalue_of_time)

padding_value = 1
padding_method = 'post'
#X2 = [np.array(i)/float(maxvalue_of_time) for i in timelist]
#X2 = [1/(np.array(i) +1) for i in timelist]

#
def newrange(x):
    nrg = 1
    nrm= 0
    org = float(maxvalue_of_time)
    return [(((float(maxvalue_of_time)-np.array(i))*float(nrg))/org + nrm) for i in x]



#def newrange(x):
#    nrg = 0.5
#    org = float(maxvalue_of_time)
#    return [(((float(maxvalue_of_time)-np.array(i))*float(nrg))/org + 0.5) for i in x]

X2 = newrange(timelist)
#X2 = [(1/(2**np.array(i))) for i in timelist]

#for index, item in enumerate(timelist):
#    for j,it in enumerate(item):
#        timelist[index][j] = (maxvalue_of_time - it)/ float(maxvalue_of_time)
X1 = sequence.pad_sequences(tplist, maxlen=max_tps_length, padding = padding_method)
X2 = sequence.pad_sequences(X2, maxlen=max_tps_length, dtype='float', padding = padding_method, value = padding_value)
X3 = sequence.pad_sequences(tpplist, maxlen=max_tps_length, padding = padding_method)

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#timelist = sc.fit_transform(timelist)


#X1 =[]
#X2 =[]
#for index, item in enumerate(tplist):
#    X1.append(tplist[index][0:10])
#    X2.append(timelist[index][0:10])
#
#
#maxnumber_of_tp = max([max(i) for i in X1]) + 1
#print(maxnumber_of_tp)
#max_tps_length = max([len(i) for i in X1])
#print(max_tps_length)
#maxvalue_of_time = max([max(i) for i in X2]) + 1
#print(maxvalue_of_time)
#
#X2 = [(float(maxvalue_of_time) - np.array(i))/ float(maxvalue_of_time) for i in X2]
#
#X1 = sequence.pad_sequences(X1, maxlen=max_tps_length)
#X2 = sequence.pad_sequences(X2, maxlen=max_tps_length)


#X2 = sequence.pad_sequences(timelist, maxlen=20, dtype='float', value = -1)

#X2 = X2/maxvalue_of_time

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X2 = sc.fit_transform(X2)


X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, y_train, y_test = train_test_split(X1, X2, X3, y, test_size = 0.2, random_state = 0)

X1_train, X1_v, X2_train, X2_v, X3_train, X3_v, y_train, y_v = train_test_split(X1_train, X2_train, X3_train, y_train, test_size = 0.2, random_state = 0)
print(len(y_train))

sum1= 0
sum2= 0
for i, item in enumerate(X1_train):
    if y_train[i] == 1:
        sum1 +=1
    if y_train[i] == 0:
        sum2 +=1
print("positive sample:", sum1)
print("negative sample:", sum2)

from keras import optimizers

opt = optimizers.Adam(lr=1e-3, decay= 1e-4)

embedding_vecor_length = 32
i11 = Input(shape = (max_tps_length,))
i12 = Embedding(maxnumber_of_tp, embedding_vecor_length, input_length = max_tps_length)(i11)
#i12 = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(i12)
#i12 = LSTM(int(i12.shape[2]), return_sequences = True)(i12)
#i12 = LSTM(int(i12.shape[2]), return_sequences = True)(i12)
i14 = LSTM(int(i12.shape[2]))(i12)


i21 = Input(shape = (max_tps_length,))
i22 = Reshape((max_tps_length,1))(i21)
#i22 = TimeDistributed(Dense(int(i13.shape[2]), activation='relu'))(i22)
i23 = Dense(32, activation='relu')(i21)

i31 = Input(shape = (max_tps_length,))
i32 = Dense(32, activation='relu')(i31)
#i32 = Reshape((max_tps_length,1))(i21)

# decoder model
#multimodal = concatenate([i13, i22], axis = -1)
##multimodal = multiply([i14, i22])
#multimodal = LSTM(100)(multimodal)
#multimodal = Flatten()(multimodal)
multimodal = concatenate([i14, i21, i21, i31], axis = -1)
#multimodal = Dense(64, activation='relu')(multimodal)
#multimodal = Dropout(0.2)(multimodal)
#multimodal = Dense(64, activation='relu')(multimodal)
#multimodal = Dropout(0.2)(multimodal)
outputs = Dense(1, activation='sigmoid')(multimodal)
# tie it together [article, summary] [word]
model = Model(inputs=[i11, i21, i31], outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])
print(model.summary())
history = model.fit([X1_train,X2_train, X3_train], y_train, validation_data=([X1_v,X2_v, X3_v], y_v), epochs= 3, batch_size=64, verbose = 1)
scores = model.evaluate([X1_test,X2_test,X3_test], y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))





idd = 1764
x11 = sequence.pad_sequences([[idd]], padding = padding_method, maxlen=max_tps_length)
xaxis = []
yaxis = []
for i in range(0, maxvalue_of_time, 1000):
#    print(i)
    x22 = newrange([[i]])
    x22 = sequence.pad_sequences(x22, maxlen=max_tps_length, dtype='float', padding = padding_method, value = padding_value)
    x33 = sequence.pad_sequences([[1]], padding = padding_method, maxlen=max_tps_length)
    yp = model.predict([x11,x22,x33])
    xaxis.append(i)
    yaxis.append(float(yp))
from matplotlib import pyplot as plt
plt.plot(xaxis, yaxis)
plt.ylim((0,1))








from keras.utils import plot_model
plot_model(model, to_file='model.png')

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
y_pred = model.predict([X1_test,X2_test])
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)


threshold= 0.5
y_pred1 = (y_pred > threshold).astype(int)
cm = confusion_matrix(y_test, y_pred1)
print('RNN cm:{}'.format(cm))
auc = roc_auc_score(y_test, y_pred1)
accuracy = accuracy_score(y_test, y_pred1)
print('RNN auc:{}'.format(auc))









x22 = [[0], [10000],[20000],[maxvalue_of_time]]
x22 = newrange(x22)
#x11 = sequence.pad_sequences([[1764],[1764], [1764],[1764]], padding = padding_method, maxlen=max_tps_length)
x11 = sequence.pad_sequences([[22],[22], [22],[22]], padding = padding_method, maxlen=max_tps_length)
x22 = sequence.pad_sequences(x22, maxlen=max_tps_length, dtype='float', padding = padding_method, value = padding_value)
x33 = sequence.pad_sequences([[1],[1], [1],[1]], padding = padding_method, maxlen=max_tps_length)
yp = model.predict([x11,x22,x33])
yp



x22 = [[0], [10000],[20000],[maxvalue_of_time]]
x22 = newrange(x22)
#x22 = [1/(np.array(i)+1) for i in x22]
#x11 = sequence.pad_sequences([[2753],[2753], [2753],[2753]], padding = padding_method, maxlen=max_tps_length)
x11 = sequence.pad_sequences([[2802],[2802], [2802],[2802]], padding = padding_method, maxlen=max_tps_length)
#x11 = sequence.pad_sequences([[2259],[2259], [2259],[2259]], padding = padding_method, maxlen=max_tps_length)
#x11 = sequence.pad_sequences([[223],[223], [223],[223]], padding = padding_method, maxlen=max_tps_length)
x22 = sequence.pad_sequences(x22, maxlen=max_tps_length, dtype='float', padding = padding_method, value =padding_value)
x33 = sequence.pad_sequences([[1],[1], [1],[1]], padding = padding_method, maxlen=max_tps_length)
yp = model.predict([x11,x22,x33])
yp



x22 = [[0, 1000], [10000,20000],[20000,20000],[maxvalue_of_time,maxvalue_of_time]]
x22 = newrange(x22)
x11 = sequence.pad_sequences([[22, 2753],[22, 2753], [22, 2753],[22, 2753]], padding = padding_method, maxlen=max_tps_length)
#x11 = sequence.pad_sequences([[411, 2536],[411, 2536], [411, 2536],[411, 2536]], padding = padding_method, maxlen=max_tps_length)
x22 = sequence.pad_sequences(x22, maxlen=max_tps_length, dtype='float', padding = padding_method, value = padding_value)
x33 = sequence.pad_sequences([[1,1],[1,1], [1,1],[1,1]], padding = padding_method, maxlen=max_tps_length)
yp = model.predict([x11,x22,x33])
yp


x22 = [[0, 5000], [10000,15000],[20000,30000],[40000,maxvalue_of_time]]
x22 = newrange(x22)
#x22 = [1/(np.array(i)+1) for i in x22]
x11 = sequence.pad_sequences([[2753, 2536],[2753, 2536], [2753, 2536],[2753, 2536]], padding = padding_method, maxlen=max_tps_length)
x22 = sequence.pad_sequences(x22, maxlen=max_tps_length, dtype='float', padding = padding_method,value = padding_value)
x33 = sequence.pad_sequences([[1,1],[1,1], [1,1],[1,1]], padding = padding_method, maxlen=max_tps_length)
yp = model.predict([x11,x22,x33])
yp
