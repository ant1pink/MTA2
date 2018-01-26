# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:52:34 2017

@author: ruili2
"""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Concatenate, Reshape, RepeatVector
from keras.layers import LSTM
from keras import layers
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
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

X1 = sequence.pad_sequences(tplist, maxlen=max_tps_length)
X2 = sequence.pad_sequences(timelist, maxlen=max_tps_length)

X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1,X2, y, test_size = 0.2, random_state = 0)


embedding_vecor_length = 32
input1 = Input(shape = (max_tps_length,))
l1 = Embedding(maxnumber_of_tp, embedding_vecor_length, input_length = max_tps_length)(input1)
l2 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(l1)
l3 = MaxPooling1D(pool_size=2)(l2)
l4 = LSTM(100)(l3)

input2 = Input(shape = (max_tps_length,))
# decoder model
decoder1 = keras.layers.Concatenate()([l4, input2])
decoder2 = Reshape((100+max_tps_length,1))(decoder1)
decoder3 = LSTM(100)(decoder2)
outputs = Dense(1, activation='sigmoid')(decoder3)
# tie it together [article, summary] [word]
model = Model(inputs=[input1, input2], outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit([X1_train,X2_train], y_train, validation_data=([X1_test,X2_test], y_test), epochs= 3, batch_size=64, verbose = 1)
scores = model.evaluate([X1_test,X2_test], y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
y_pred = model.predict([X1_test,X2_test])
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)

#####################################################################


embedding_vecor_length = 32
input1 = Input(shape = (max_tps_length,))
l1 = Embedding(maxnumber_of_tp, embedding_vecor_length, input_length = max_tps_length)(input1)
l2 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(l1)
l3 = MaxPooling1D(pool_size=2)(l2)
l4 = LSTM(100)(l3)





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

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.3, random_state = 0)

# RNN model
embedding_vecor_length = 32
optimizer = optimizers.Adam(lr=0.01, decay=1e-3)
model = Sequential()
model.add(Embedding(maxnumber_of_tp, embedding_vecor_length, input_length=max_tps_length))

model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))


model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs= 3, batch_size=64, verbose = 1)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# CNN model
from keras.layers import Dropout
embedding_vecor_length = 32
optimizer = optimizers.Adam(lr=0.01, decay=1e-3)
model = Sequential()
model.add(Embedding(maxnumber_of_tp, embedding_vecor_length, input_length=max_tps_length))
model.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs= 50, batch_size=32, verbose = 1)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



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

xp = sequence.pad_sequences([[81,100],[100,81]], maxlen=max_tps_length)
yp = model.predict(xp)

