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

with open('./22012018/user_tplist.pkl', 'rb') as fid:
    tplist = pickle.load(fid)
with open('./22012018/user_tptimelist.pkl', 'rb') as fid:
    timelist = pickle.load(fid)
with open('./22012018/label.pkl', 'rb') as fid:
    y = pickle.load(fid)

maxnumber_of_tp = max([max(i) for i in tplist]) + 1
print(maxnumber_of_tp)
max_tps_length = max([len(i) for i in tplist])
print(max_tps_length)
maxvalue_of_time = max([max(i) for i in timelist]) + 1
print(maxvalue_of_time)

padding_value = 0
#X2 = [np.array(i)/float(maxvalue_of_time) for i in timelist]
#X2 = [1/(np.array(i) +1) for i in timelist]

X2 = [(float(maxvalue_of_time) - np.array(i))/ float(maxvalue_of_time) for i in timelist]

X1 = sequence.pad_sequences(tplist, maxlen=max_tps_length)
X2 = sequence.pad_sequences(X2, maxlen=max_tps_length, dtype='float', value = padding_value)


#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#timelist = sc.fit_transform(timelist)


#X1 =[]
#X2 =[]
#for index, item in enumerate(tplist):
#    X1.append(tplist[index][0:20])
#    X2.append(timelist[index][0:20])
#
#
#maxnumber_of_tp = max([max(i) for i in X1]) + 1
#print(maxnumber_of_tp)
#max_tps_length = max([len(i) for i in X1])
#print(max_tps_length)
#maxvalue_of_time = max([max(i) for i in X2]) + 1
#print(maxvalue_of_time)
#
#
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

opt = optimizers.Adam(lr=1e-3, decay= 1e-4)

embedding_vecor_length = 32
i11 = Input(shape = (max_tps_length,))
i12 = Embedding(maxnumber_of_tp, embedding_vecor_length, input_length = max_tps_length)(i11)
i13 = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(i12)

i14 = LSTM(100, return_sequences = True)(i13)
i15 = Dropout(0.2)(i14)
i16 = LSTM(100)(i15)
i17 = Dropout(0.2)(i16)


i21 = Input(shape = (max_tps_length,))
i23 = Dense(32, activation='relu')(i21)
#i23 = Dropout(0.2)(i23)
i22 = Reshape((max_tps_length,1))(i21)
#i22 = MaxPooling1D(pool_size=2)(i22)
#i23 = Flatten()(i22)
#i23 = TimeDistributed(Dense(64,  activation='relu'))(i22)

# decoder model
multimodal = multiply([i15, i22])
multimodal = LSTM(100)(multimodal)
multimodal = Dropout(0.2)(multimodal)
#multimodal = Flatten()(multimodal)
multimodal = concatenate([i17, multimodal, i23])
multimodal = Dense(128, activation='relu')(multimodal)
multimodal = Dropout(0.2)(multimodal)
multimodal = Dense(128, activation='relu')(multimodal)
multimodal = Dropout(0.2)(multimodal)
outputs = Dense(1, activation='sigmoid')(multimodal)
# tie it together [article, summary] [word]
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


#embedding_vecor_length = 32
#model = Sequential()
#model.add(Embedding(maxnumber_of_tp, embedding_vecor_length, input_length=max_tps_length))
#model.add(Flatten())
#model.compile('rmsprop', 'mse')
#X11=  model.predict(X1)
#
#
#for i, j in enumerate(X1):
#    X1[i] = X1[i] + X2[i]





#history
import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
threshold= 0.5
y_pred1 = (y_pred > threshold).astype(int)
cm = confusion_matrix(y_test, y_pred1)
print(cm)


x22 = [[0], [10000],[20000],[maxvalue_of_time]]
x22 = [(float(maxvalue_of_time) - np.array(i))/ float(maxvalue_of_time) for i in x22]
x11 = sequence.pad_sequences([[951],[951], [951],[951]], maxlen=max_tps_length)
x22 = sequence.pad_sequences(x22, maxlen=max_tps_length, dtype='float', value = padding_value)
yp = model.predict([x11,x22])
yp



x22 = [[0], [10000],[20000],[maxvalue_of_time]]
x22 = [(float(maxvalue_of_time) - np.array(i))/ float(maxvalue_of_time) for i in x22]
#x22 = [1/(np.array(i)+1) for i in x22]
#x11 = sequence.pad_sequences([[951],[951], [951],[951]], maxlen=max_tps_length)
#x11 = sequence.pad_sequences([[1219],[1219], [1219],[1219]], maxlen=max_tps_length)
x11 = sequence.pad_sequences([[1230],[1230], [1230],[1230]], maxlen=max_tps_length)
#x11 = sequence.pad_sequences([[1213],[1213], [1213],[1213]], maxlen=max_tps_length)
x22 = sequence.pad_sequences(x22, maxlen=max_tps_length, dtype='float', value =padding_value)
yp = model.predict([x11,x22])
yp



x22 = [[5000, 0], [20000,10000],[30000,20000],[maxvalue_of_time,maxvalue_of_time]]
x22 = [(float(maxvalue_of_time) - np.array(i))/ float(maxvalue_of_time) for i in x22]
x11 = sequence.pad_sequences([[1230, 951],[1230, 951], [1230, 951],[1230, 951]], maxlen=max_tps_length)
x22 = sequence.pad_sequences(x22, maxlen=max_tps_length, dtype='float', value = padding_value)
yp = model.predict([x11,x22])
yp


x22 = [[5000, 0], [20000,15000],[40000,30000],[maxvalue_of_time,40000]]
x22 = [(float(maxvalue_of_time) - np.array(i))/ float(maxvalue_of_time) for i in x22]
#x22 = [1/(np.array(i)+1) for i in x22]
x11 = sequence.pad_sequences([[1230, 1219],[1230, 1219], [1230, 1219],[1230, 1219]], maxlen=max_tps_length)
x22 = sequence.pad_sequences(x22, maxlen=max_tps_length, dtype='float', value = padding_value)
yp = model.predict([x11,x22])
yp
