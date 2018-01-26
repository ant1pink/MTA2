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

for index, item in enumerate(timelist):
    for i,j in enumerate(item):
        time_v = j/float(maxvalue_of_time)
#        print(time_v)
        if time_v < 0.1:
            timelist[index][i] = 1
        elif time_v >= 0.1 and time_v < 0.2:
            timelist[index][i] = 2
        elif time_v >= 0.2 and time_v < 0.3:
            timelist[index][i] = 3
        elif time_v >= 0.3 and time_v < 0.4:
            timelist[index][i] = 4
        elif time_v >= 0.4 and time_v < 0.5:
            timelist[index][i] = 5
        elif time_v >= 0.5 and time_v < 0.6:
            timelist[index][i] = 6
        elif time_v >= 0.6 and time_v < 0.7:
            timelist[index][i] = 7
        elif time_v >= 0.7 and time_v < 0.8:
            timelist[index][i] = 8
        elif time_v >= 0.8 and time_v < 0.9:
            timelist[index][i] = 9
        elif time_v >= 0.9 and time_v < 1:
            timelist[index][i] = 10
#X2 = [(np.array(i) - (maxvalue_of_time/2))/maxvalue_of_time for i in timelist]

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#timelist = sc.fit_transform(timelist)


X1 = sequence.pad_sequences(tplist, maxlen=max_tps_length)
#X2 = sequence.pad_sequences(timelist, maxlen=max_tps_length, dtype='float', value = -1)
X2 = sequence.pad_sequences(timelist, maxlen=max_tps_length)
#X2 = X2/maxvalue_of_time

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X2 = sc.fit_transform(X2)


X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1,X2, y, test_size = 0.2, random_state = 0)
X1_train, X1_v, X2_train, X2_v, y_train, y_v = train_test_split(X1_train, X2_train, y_train, test_size = 0.2, random_state = 1)

from keras import optimizers

opt = optimizers.Adam(lr=1e-3)

embedding_vecor_length = 32
i11 = Input(shape = (max_tps_length,))
i12 = Embedding(maxnumber_of_tp, embedding_vecor_length, input_length = max_tps_length)(i11)
i13 = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(i12)
i14 = MaxPooling1D(pool_size=2)(i13)
i15 = LSTM(100)(i14)
i16 = Dropout(0.2)(i15)
#i17 = LSTM(100, return_sequences = True)(i16)
#i18 = Dropout(0.2)(i17)
#i19 = LSTM(100)(i18)
#i20 = Dropout(0.2)(i19)
#i15 = Bidirectional(LSTM(32))(i14)
#i16 = Bidirectional(LSTM(32))(i15)
#i14 = LSTM(100, return_sequences = True, input_shape = i13.shape)(i13)


i21 = Input(shape = (max_tps_length,))
i22 = Embedding(11, embedding_vecor_length, input_length = max_tps_length)(i21)
i23 = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(i22)
i24 = MaxPooling1D(pool_size=2)(i23)
i25 = LSTM(100)(i24)
i26 = Dropout(0.2)(i25)
#i27 = LSTM(100, return_sequences = True)(i26)
#i28 = Dropout(0.2)(i27)
#i29 = LSTM(100)(i28)
#i30 = Dropout(0.2)(i29)
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
multimodal = concatenate([i14, i24], axis=-1)
multimodal = LSTM(100, return_sequences = True)(multimodal)
multimodal = Dropout(0.2)(multimodal)
multimodal = LSTM(100)(multimodal)
multimodal = Dropout(0.2)(multimodal)
#multimodal = Dense(64, activation='relu')(multimodal)
#multimodal = Dropout(0.2)(multimodal)
#multimodal = Dense(64, activation='relu')(multimodal)
#multimodal = Dropout(0.2)(multimodal)
#multimodal = Dense(64, activation='relu')(multimodal)
#decoder3 = Dense(100, activation='relu')(decoder1)

#decoder3 = LSTM(100)(decoder1)
multimodal = concatenate([i16, i26, multimodal], axis=-1)
multimodal = Dense(64, activation='relu')(multimodal)
multimodal = Dropout(0.2)(multimodal)
multimodal = Dense(64, activation='relu')(multimodal)
multimodal = Dropout(0.2)(multimodal)
outputs = Dense(1, activation='sigmoid')(multimodal)
# tie it together [article, summary] [word]
model = Model(inputs=[i11, i21], outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])
print(model.summary())

#history = model.fit([X1_train,X2_train], y_train, validation_data=([X1_test,X2_test], y_test), epochs= 3, batch_size=64, verbose = 1)
history = model.fit([X1_train,X2_train], y_train, validation_data=([X1_v,X2_v], y_v), epochs= 5, batch_size=32, verbose = 1)
scores = model.evaluate([X1_test,X2_test], y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
y_pred = model.predict([X1_test,X2_test])
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)





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


x22 = [[9,0], [9,3],[9,8],[9],[6],[3],[0]]
#x22 = [np.array(i)/maxvalue_of_time for i in x22]
x11 = sequence.pad_sequences([[1213, 1163],[1213, 1163], [1213, 1163], [1163], [1163], [1163], [1163]], maxlen=max_tps_length)
x22 = sequence.pad_sequences(x22, maxlen=max_tps_length)
#x22 = sc.transform(x22)
#x22=x22/maxvalue_of_time
yp = model.predict([x11,x22])
yp
