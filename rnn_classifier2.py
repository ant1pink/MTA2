# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:52:34 2017

@author: ruili2
"""
from __future__ import division
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input, concatenate, Reshape, RepeatVector, TimeDistributed, Bidirectional, Flatten, Dropout
from keras.layers import LSTM
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

with open('./19012018/user_tplist.pkl', 'rb') as fid:
    tplist = pickle.load(fid)
with open('./19012018/user_tptimelist.pkl', 'rb') as fid:
    timelist = pickle.load(fid)
with open('./19012018/label.pkl', 'rb') as fid:
    y = pickle.load(fid)

maxnumber_of_tp = max([max(i) for i in tplist]) + 1
print(maxnumber_of_tp)
max_tps_length = max([len(i) for i in tplist])
print(max_tps_length)

maxvalue_of_time = max([max(i) for i in timelist]) + 1
print(maxvalue_of_time)


#X2 = [np.array(i)/float(maxvalue_of_time) for i in timelist]

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#timelist = sc.fit_transform(timelist)
#X1 =[]
#X2 =[]
#for index, item in enumerate(tplist):
#    X1.append(tplist[index][0:19])
#    X2.append(timelist[index][0:19])
#max_tps_length = 20

X1 = sequence.pad_sequences(tplist, maxlen=max_tps_length, value = 0)
X2 = sequence.pad_sequences(timelist, maxlen=max_tps_length, value = 0)
#X2 = sequence.pad_sequences(timelist, maxlen=20, dtype='float', value = -1)

#X2 = X2/maxvalue_of_time

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X2 = sc.fit_transform(X2)


X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size = 0.2, random_state = 1)

X1_train, X1_v, X2_train, X2_v, y_train, y_v = train_test_split(X1_train, X2_train, y_train, test_size = 0.2, random_state = 1)


from keras import optimizers

opt = optimizers.Adam(lr=1e-3, decay = 1e-4)

embedding_vecor_length = 32
i11 = Input(shape = (max_tps_length,))
i12 = Embedding(maxnumber_of_tp, embedding_vecor_length, input_length = max_tps_length)(i11)
i13 = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(i12)
#i14 = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(i13)
i14 = MaxPooling1D(pool_size=2)(i13)
#i15 = Bidirectional(LSTM(100, return_sequences = True))(i14)
#i16 = Bidirectional(LSTM(100))(i15)
#i15 = LSTM(100, return_sequences = True)(i13)
#i16 = Dropout(0.2)(i15)
i17 = LSTM(100)(i14)
#i18 = Dropout(0.2)(i17)
#i19 = LSTM(100)(i18)
#i20 = Dropout(0.2)(i19)
#i15 = Bidirectional(LSTM(32))(i14)
#i16 = Bidirectional(LSTM(32))(i15)
#i14 = LSTM(100, return_sequences = True, input_shape = i13.shape)(i13)


i21 = Input(shape = (max_tps_length,))
#i22 = Reshape((max_tps_length,1))(i21)

#i21 = Input(shape = (None, max_tps_length))
#i22 = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(i21)
#i23 = MaxPooling1D(pool_size=2)(i22)

#i22 = Reshape((max_tps_length,1))(i21)
#i23 = LSTM(50, return_sequences = True)(i22)
#i24 = Dropout(0.2)(i23)
#i25 = LSTM(50, return_sequences = True)(i24)
#i26 = Dropout(0.2)(i25)
#i27 = LSTM(50)(i26)
#i28 = Dropout(0.2)(i27)


#i23 = TimeDistributed(Dense(64,  activation='relu'))(i22)


#i22 = Embedding(maxvalue_of_time, embedding_vecor_length, input_length = max_tps_length)(i21)
#i23 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(i22)
#i24 = MaxPooling1D(pool_size=2)(i23)
#i22 = Reshape((max_tps_length,1))(i21)
#i22 = RepeatVector(max_tps_length)(i21) # better timedecay
# decoder model
multimodal = concatenate([i17, i21], axis=-1)
#multimodal = Dense(60, activation='relu')(multimodal)
#multimodal = Dropout(0.2)(multimodal)
#multimodal = Dense(64, activation='relu')(multimodal)
#multimodal = Dropout(0.2)(multimodal)
#multimodal = Dense(64, activation='relu')(multimodal)
#decoder3 = Dense(100, activation='relu')(decoder1)

#decoder3 = LSTM(100)(decoder1)

outputs = Dense(1, activation='sigmoid')(multimodal)
# tie it together [article, summary] [word]
model = Model(inputs=[i11, i21], outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer= 'Adam', metrics=['accuracy'])
print(model.summary())

#history = model.fit([X1_train,X2_train], y_train, validation_data=([X1_test,X2_test], y_test), epochs= 3, batch_size=64, verbose = 1)
history = model.fit([X1_train,X2_train], y_train, validation_data=([X1_v,X2_v], y_test), epochs= 15, batch_size=64, verbose = 1)
scores = model.evaluate([X1_test,X2_test], y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


from keras.utils import plot_model
plot_model(model, to_file='model.png')



from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
y_pred = model.predict([X1_test,X2_test])
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


x22 = [[10], [100000],[1],[100],[20000`0],[100],[0]]
x22 = [np.array(i)/float(maxvalue_of_time) for i in x22]
x11 = sequence.pad_sequences([[2911],[2911], [122],[122],[122], [122], [882]], maxlen=max_tps_length)
x22 = sequence.pad_sequences(x22, maxlen=max_tps_length, dtype='float', value =-1)
#x22 = sc.transform(x22)
#x22=x22/maxvalue_of_time
yp = model.predict([x11,x22])
yp
