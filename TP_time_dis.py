# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 15:42:26 2018

@author: ruili2
"""
from __future__ import division
import pickle
from matplotlib import pyplot as plt
import numpy as np


with open('./23012018/user_tplist.pkl', 'rb') as fid:
    tplist = pickle.load(fid)
with open('./23012018/user_tptimelist.pkl', 'rb') as fid:
    timelist = pickle.load(fid)
with open('./23012018/label.pkl', 'rb') as fid:
    y = pickle.load(fid)
    
    
maxvalue_of_time = max([max(i) for i in timelist]) + 1
print(maxvalue_of_time)


#for index, item in enumerate(timelist):
#    for i,j in enumerate(item):
#        time_v = j/float(maxvalue_of_time)
##        print(time_v)
#        if time_v < 0.1:
#            timelist[index][i] = 1
#        elif time_v >= 0.1 and time_v < 0.2:
#            timelist[index][i] = 2
#        elif time_v >= 0.2 and time_v < 0.3:
#            timelist[index][i] = 3
#        elif time_v >= 0.3 and time_v < 0.4:
#            timelist[index][i] = 4
#        elif time_v >= 0.4 and time_v < 0.5:
#            timelist[index][i] = 5
#        elif time_v >= 0.5 and time_v < 0.6:
#            timelist[index][i] = 6
#        elif time_v >= 0.6 and time_v < 0.7:
#            timelist[index][i] = 7
#        elif time_v >= 0.7 and time_v < 0.8:
#            timelist[index][i] = 8
#        elif time_v >= 0.8 and time_v < 0.9:
#            timelist[index][i] = 9
#        elif time_v >= 0.9 and time_v < 1:
#            timelist[index][i] = 10

sum1= []
sum2= 0
tid = 121
for i, item in enumerate(tplist):
    for j, item2 in enumerate(item):
        if item2 == tid and y[i] == 1:
##        if y[i] == 1:
#            print(item2)
#            print(timelist[i][j])
            sum1.append(timelist[i][j])
            sum2 +=1
            
a = [[x, sum1.count(x)] for x in set(sum1)]
a.sort()
b = np.array(a)
print(tid, 'sum2 is', sum2)
#plt.scatter(b[:,0],b[:, 1])
plt.hist(sum1)
plt.show()