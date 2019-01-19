import sys
import numpy as np
import math
from sklearn import svm
from sklearn.feature_selection import RFE
import scipy.stats as stats
import xgboost as xgb
import matplotlib.pyplot as plt
import graphviz
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import pickle

data_path = 'preprocess/f_240_CRE_v1.csv'
data_all = np.loadtxt(data_path, dtype = 'float', delimiter = ',')

data_num = data_all.shape[0]
sub_num = 33
sti_num = data_num//sub_num

fn = 240
f_x = data_all[:,0:fn]
a_y_temp = data_all[:,fn]
v_y_temp = data_all[:,fn+1]
a_y = (a_y_temp>4.9999)*1
v_y = (v_y_temp>4.9999)*1
print ( f_x.shape)

a_y = np.zeros((data_num,))
v_y = np.zeros((data_num,))
for i in range(data_num):
    j= i//sti_num
    me_a = np.mean(a_y_temp[j*sti_num:j*sti_num+sti_num])
    me_v = np.mean(v_y_temp[j*sti_num:j*sti_num+sti_num])
    a_y[i] =  (a_y_temp[i]>(me_a))*1
    v_y[i] =  (v_y_temp[i]>(me_v))*1

v_x = f_x
a_x = f_x

print ( '----------------------------------------------------------')
print ( 'Valence:')
Ein_v = np.zeros((sub_num,))
Eval_v = np.zeros((sub_num,))
y_pred_v = np.zeros((data_num,))
y_pred_v_value = np.zeros((data_num,))


for i in range(sub_num):
	#print ( "sub_num = %i" %(i+1))
	k = i*sti_num
	train_index = list(range(0, k)) + list(range(k+16, data_num))
	test_index = list(range(k, k+16))
	#print ( train_index
	#print ( test_index
	x_train, x_test = v_x[train_index], v_x[test_index]
	y_train, y_test = v_y[train_index], v_y[test_index]

	clf_v = xgb.XGBRegressor(n_estimators=10, max_depth=3, learning_rate=0.15, reg_lambda=2)
	clf_v.fit(x_train, y_train)	
	y_pred_value = clf_v.predict(x_test)
	y_pred = (clf_v.predict(x_test)>0.5)*1
	y_predin = (clf_v.predict(x_train)>0.5)*1
	
	count_in =  sum(y_predin == y_train)
	count_val =  sum(y_pred == y_test)
	#y_pred_a[i] = y_pred
	Ein_v[i] =  float(count_in)/len(y_train)
	Eval_v[i] =  float(count_val)/len(y_test)
	y_pred_v[i*sti_num:i*sti_num+sti_num] = y_pred
	y_pred_v_value[i*sti_num:i*sti_num+sti_num] = y_pred_value

print ( "f1_score = %f" %(f1_score(v_y, y_pred_v, average='macro')) )
print ( "acc_in = %f" %(Ein_v.mean()))
print ( "acc_val = %f" %(Eval_v.mean()))
print ( '----------------------------------------------------------')


print ( 'Arousal:')


Ein_a = np.zeros((sub_num,))
Eval_a = np.zeros((sub_num,))
y_pred_a = np.zeros((data_num,))
y_pred_a_value = np.zeros((data_num,))

for i in range(sub_num):
	k = i*sti_num
	train_index = list(range(0, k)) + list(range(k+sti_num, data_num))
	test_index = list(range(k, k+sti_num))
	#print(train_index)
	#print(test_index)
	x_train, x_test = a_x[train_index], a_x[test_index]
	y_train, y_test = a_y[train_index], a_y[test_index]
		
	clf_a = xgb.XGBRegressor(n_estimators=9, max_depth=2, learning_rate=0.1, reg_lambda=2)
	clf_a.fit(x_train, y_train)
	y_pred_value = clf_a.predict(x_test)
	y_pred = (clf_a.predict(x_test)>0.5)*1
	y_predin = (clf_a.predict(x_train)>0.5)*1
	
	count_in =  sum(y_predin == y_train)
	count_val =  sum(y_pred == y_test)
	Ein_a[i] =  float(count_in)/len(y_train)
	Eval_a[i] =  float(count_val)/len(y_test)
	y_pred_a[i*sti_num:i*sti_num+sti_num] = y_pred
	y_pred_a_value[i*sti_num:i*sti_num+sti_num] = y_pred_value
	
print ( "f1_score = %f" %(f1_score(a_y, y_pred_a, average='binary')))
print ( "acc_in = %f" %(Ein_a.mean()))
print ( "acc_val = %f" %(Eval_a.mean()))
print ( '----------------------------------------------------------')
