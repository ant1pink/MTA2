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

with open('./23012018/user_tplist.pkl', 'rb') as fid:
    tplist = pickle.load(fid)
with open('./23012018/user_tptimelist.pkl', 'rb') as fid:
    timelist = pickle.load(fid)
with open('./23012018/label.pkl', 'rb') as fid:
    y = pickle.load(fid)
  


maxnumber_of_tp = max([max(i) for i in tplist]) + 1
print(maxnumber_of_tp)
max_tps_length = max([len(i) for i in tplist])
print(max_tps_length)
maxvalue_of_time = max([max(i) for i in timelist]) + 1
print(maxvalue_of_time)

padding_value = 0
padding_method = 'post'
#X2 = [np.array(i)/float(maxvalue_of_time) for i in timelist]
#X2 = [1/(np.array(i) +1) for i in timelist]

def newrange(x):
    nrg = 0.5
    org = float(maxvalue_of_time)
    return [(((float(maxvalue_of_time)-np.array(i))*float(nrg))/org + 0.5) for i in x]

X2 = newrange(timelist)
#X2 = [(1/(2**np.array(i))) for i in timelist]

#for index, item in enumerate(timelist):
#    for j,it in enumerate(item):
#        timelist[index][j] = (maxvalue_of_time - it)/ float(maxvalue_of_time)
X1 = sequence.pad_sequences(tplist, maxlen=max_tps_length, padding = padding_method,)
X2 = sequence.pad_sequences(X2, maxlen=max_tps_length, dtype='float', padding = padding_method, value = padding_value)


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


X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size = 0.2, random_state = 0)

X1_train, X1_v, X2_train, X2_v, y_train, y_v = train_test_split(X1_train, X2_train, y_train, test_size = 0.2, random_state = 0)
print(len(y_test))

from keras import optimizers

opt = optimizers.Adam(lr=1e-4)

embedding_vecor_length = 64
i11 = Input(shape = (max_tps_length,))
i12 = Embedding(maxnumber_of_tp, embedding_vecor_length, input_length = max_tps_length)(i11)
i13 = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(i12)

i21 = Input(shape = (max_tps_length,))
i22 = Reshape((max_tps_length,1))(i21)
#i22 = TimeDistributed(Dense(64))(i22)
#i22 = Reshape((max_tps_length,1))(i21)
#l1 = dot([i12, i22], axes = -1)
l1 = multiply([i12, i22])
l1 = LSTM(64, return_sequences = True, dropout = 0.2, recurrent_dropout = 0.2)(l1)
#l1 = dot([l1, i22], axes = -1)
l1 = multiply([l1, i22])
l1 = LSTM(64, return_sequences = True, dropout = 0.2, recurrent_dropout = 0.2)(l1)
#l1 = dot([l1, i22], axes = -1)
l1 = multiply([l1, i22])
l1 = LSTM(64, dropout = 0.2, recurrent_dropout = 0.2)(l1)
outputs = Dense(1, activation='sigmoid')(l1)
model = Model(inputs=[i11, i21], outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])
print(model.summary())

#history = model.fit([X1_train,X2_train], y_train, validation_data=([X1_test,X2_test], y_test), epochs= 3, batch_size=64, verbose = 1)
history = model.fit([X1_train,X2_train], y_train, validation_data=([X1_v,X2_v], y_v), epochs= 3, batch_size=64, verbose = 1)
scores = model.evaluate([X1_test,X2_test], y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))




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
x11 = sequence.pad_sequences([[1163],[1163], [1163],[1163]], padding = padding_method, maxlen=max_tps_length)
x22 = sequence.pad_sequences(x22, maxlen=max_tps_length, dtype='float', padding = padding_method, value = padding_value)
yp = model.predict([x11,x22])
yp



x22 = [[0], [10000],[20000],[30000]]
x22 = newrange(x22)
#x22 = [1/(np.array(i)+1) for i in x22]
#x11 = sequence.pad_sequences([[2737],[2737], [2737],[2737]], padding = padding_method, maxlen=max_tps_length)
x11 = sequence.pad_sequences([[2863],[2863], [2863],[2863]], padding = padding_method, maxlen=max_tps_length)
#x11 = sequence.pad_sequences([[2536],[2536], [2536],[2536]], padding = padding_method, maxlen=max_tps_length)
#x11 = sequence.pad_sequences([[223],[223], [223],[223]], padding = padding_method, maxlen=max_tps_length)
x22 = sequence.pad_sequences(x22, maxlen=max_tps_length, dtype='float', padding = padding_method, value =padding_value)
yp = model.predict([x11,x22])
yp



x22 = [[0, 1000], [10000,20000],[20000,20000],[maxvalue_of_time,maxvalue_of_time]]
x22 = newrange(x22)
x11 = sequence.pad_sequences([[1163,2863],[1163, 2863], [1163, 2863],[1163, 2863]], padding = padding_method, maxlen=max_tps_length)
x22 = sequence.pad_sequences(x22, maxlen=max_tps_length, dtype='float', padding = padding_method, value = padding_value)
yp = model.predict([x11,x22])
yp


x22 = [[0, 5000], [10000,15000],[20000,40000],[40000,maxvalue_of_time]]
x22 = newrange(x22)
#x22 = [1/(np.array(i)+1) for i in x22]
x11 = sequence.pad_sequences([[2804, 2536],[2804, 2536], [2804, 2536],[2804, 2536]], padding = padding_method, maxlen=max_tps_length)
x22 = sequence.pad_sequences(x22, maxlen=max_tps_length, dtype='float', padding = padding_method,value = padding_value)
yp = model.predict([x11,x22])
yp
