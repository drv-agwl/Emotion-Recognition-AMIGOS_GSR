import csv
import random
from sys import argv
import warnings  
from xgb_utils2 import *
import scipy.stats as stats


"======setting======" 
threshold = 1 #different feature number
fold = 33
# participant (1-40) except 5 people
# exp (1-16)
# 0 exp_number
# 2-208 feature !!!!we don't want first feature
# 209 arousal, 210 valence, 211 dominance, 212 liking, 213 familiarity 
# 214-220 emotion
"======setting======"

def p_value(x,y):
	l_0 = []
	l_1 = []
	for i in range(len(y)):
		if(y[i]==0):
			l_0.append(x[i])
		else:
			l_1.append(x[i])
	f, p = stats.f_oneway(l_0, l_1)       
	return p

def calculateP(name, label_all, data_all,task):
	print('write '+task+' P value...')
	mean = np.mean(data_all,axis=0)
	std = np.std(data_all,axis=0)
	data_all = (data_all-mean)/std
	p = []
	for i in range(data_all.shape[1]):
		p.append(p_value(data_all[:,i],label_all))
	file = open(task+'_p_value.csv','w')
	for i in range(len(p)):
		file.write(str(i)+','+str(name[i])+','+str(p[i]))
		file.write('\n')
	file.close()

def onehot(label):
	newlabel = np.zeros(len(label)).astype('int')
	allmean = 0
	for i in range(len(label)):
		j= i//16
		mean = np.mean(label[j*16:j*16+16])
		allmean+=mean
		newlabel[i] = int((label[i]>mean)*1)
	allmean = allmean/len(label)
	return allmean, newlabel

def data_shuffle(X, Y):
	pair = list(zip(X,Y))
	random.shuffle(pair)
	X, Y = zip(*pair)
	return np.array(X), np.array(Y)

def readfile(filename):
	file = open(filename,'r')
	a_label_all = []
	v_label_all = []
	data_all = []
	alllist = list(csv.reader(file))
	for n, row in enumerate(alllist):
		a_label_all.append(float(row[213]))
		v_label_all.append(float(row[214]))
		data_all.append(row[0:32])
	a_label_all = np.array(a_label_all)
	v_label_all = np.array(v_label_all)
	data_all = np.array(data_all).astype('float')
	file.close()
	return a_label_all, v_label_all ,data_all

def writefile(filename, accuracy,task,write):
	writefile = open(filename,write)
	writefile.write(task+'\n')
	writefile.write('feature_size, train_acc, train_f1, val_acc, val_f1\n')
	for i in range(len(accuracy)):
		writefile.write(','.join(repr(accuracy[i][j]) for j in range(len(accuracy[i]))))
		writefile.write('\n')
	writefile.close()

def main():
	a_label_all, v_label_all, data_all = readfile(infile)
	#print(data_all.shape,label_all.shape)
	#input()
	"======p_value=====onehot======"
	a_cut, a_label_all = onehot(a_label_all)
	v_cut, v_label_all = onehot(v_label_all)
	#calculateP(name, a_label_all, data_all,'arousal')
	#calculateP(name, v_label_all, data_all,'valence')
	"===========variable==========="
	a_train_acc = np.zeros((threshold,fold))
	a_train_f1 = np.zeros((threshold,fold))
	a_val_f1 = np.zeros((threshold,fold))
	a_val_acc = np.zeros((threshold,fold))
	a_val_roc = np.zeros((threshold,fold))
	a_feature_size = np.zeros((threshold,fold))
	v_train_acc = np.zeros((threshold,fold))
	v_train_f1 = np.zeros((threshold,fold))
	v_val_f1 = np.zeros((threshold,fold))
	v_val_acc = np.zeros((threshold,fold))
	v_val_roc = np.zeros((threshold,fold))
	v_feature_size = np.zeros((threshold,fold))
	"======35 fold validation======"
	for val in list(range(fold)):
		#print('============================')
		print('val subject ', val)
		"===========train and val==========="
		train_index = list(range(0, val*16)) + list(range(val*16+16, len(data_all)))
		val_index = list(range(val*16, val*16+16))
		X_train = data_all[train_index]
		X_val = data_all[val_index]
		a_Y_train = a_label_all[train_index]
		a_Y_val = a_label_all[val_index]
		v_Y_train = v_label_all[train_index]
		v_Y_val = v_label_all[val_index]
		"===========normalize==========="
		mean = np.mean(X_train,axis=0)
		std = np.std(X_train,axis=0)
		X_train = (X_train-mean)/std
		X_val = (X_val-mean)/std
		"==========shuffle=========="
		a_X_train = X_train
		v_X_train = X_train
		a_X_train, a_Y_train = data_shuffle(a_X_train,a_Y_train)
		v_X_train, v_Y_train = data_shuffle(v_X_train,v_Y_train)
		"===========train==========="
		a_tn_acc, a_tn_f1, a_v_acc, a_v_f1, a_roc, a_feat_size = train(a_X_train,a_Y_train,X_val,a_Y_val, val, threshold,'arousal')
		a_train_acc[:,val] = a_tn_acc
		a_train_f1[:,val] = a_tn_f1
		a_val_acc[:,val] = a_v_acc
		a_val_f1[:,val] = a_v_f1
		a_val_roc[:,val] = a_roc
		a_feature_size[:,val] = a_feat_size
		v_tn_acc, v_tn_f1, v_v_acc, v_v_f1, v_roc, v_feat_size = train(v_X_train,v_Y_train,X_val,v_Y_val, val, threshold,'valence')
		v_train_acc[:,val] = v_tn_acc
		v_train_f1[:,val] = v_tn_f1
		v_val_acc[:,val] = v_v_acc
		v_val_f1[:,val] = v_v_f1
		v_val_roc[:,val] = v_roc
		v_feature_size[:,val] = v_feat_size
		#cutline.append(cut)
		#one.append(onetrain+oneval)
		#zero.append(zerotrain+zeroval)

	"===========getresult==========="
	#allonezero = sum(one)+sum(zero)	
	#one = sum(one)/allonezero
	#zero = sum(zero)/allonezero
	#cutline = np.mean(np.array(cutline))
	a_train_acc = np.sum(a_train_acc,axis=1)/(fold)
	a_train_f1 = np.sum(a_train_f1,axis=1)/(fold)
	a_val_acc = np.sum(a_val_acc,axis=1)/(fold)
	a_val_f1 = np.sum(a_val_f1,axis=1)/(fold)
	a_val_roc = np.sum(a_val_roc,axis=1)/(fold)
	a_feature_size = np.sum(a_feature_size,axis=1)/(fold)

	v_train_acc = np.sum(v_train_acc,axis=1)/(fold)
	v_train_f1 = np.sum(v_train_f1,axis=1)/(fold)
	v_val_acc = np.sum(v_val_acc,axis=1)/(fold)
	v_val_f1 = np.sum(v_val_f1,axis=1)/(fold)
	v_val_roc = np.sum(v_val_roc,axis=1)/(fold)
	v_feature_size = np.sum(v_feature_size,axis=1)/(fold)
	
	a_one = len(np.where(a_label_all==int(1))[0])/len(a_label_all)
	v_one = len(np.where(v_label_all==int(1))[0])/len(v_label_all)
	print("===============================")
	print('arousal:') 
	print('one: %2f zero: %2f cutline: %2f' % (a_one,1-a_one,a_cut))
	print('feature_size = '+str(a_feature_size))
	print('train accuracy = '+str(a_train_acc))
	print('train f1 score = '+str(a_train_f1))
	print('test accuracy = '+str(a_val_acc))
	print('test f1 score = '+str(a_val_f1))
	print('test roc score = '+str(a_val_roc))
	getbestfeature('arousal')
	print("===============================")
	print('valence:')
	print('one: %2f zero: %2f cutline: %2f' % (v_one,1-v_one,v_cut))
	print('feature_size = '+str(v_feature_size))
	print('train accuracy = '+str(v_train_acc))
	print('train f1 score = '+str(v_train_f1))
	print('test accuracy = '+str(v_val_acc))
	print('test f1 score = '+str(v_val_f1))
	print('test roc score = '+str(v_val_roc))
	getbestfeature('valence')
	a_accuracy = np.array([a_feature_size, a_train_acc, a_train_f1, a_val_acc, a_val_f1, a_val_roc]).T
	v_accuracy = np.array([v_feature_size, v_train_acc, v_train_f1, v_val_acc, v_val_f1, v_val_roc]).T
	writefile(outfile,a_accuracy,'arousal','w')
	writefile(outfile,v_accuracy,'valence','a')


if __name__ == '__main__':
	warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)  
	infile = argv[1]
	outfile = argv[2]
	main()

