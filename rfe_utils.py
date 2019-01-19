import numpy as np 
import os 
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC,LinearSVC
from sklearn.feature_selection import RFE, RFECV
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

collectnum = 1

def getbestfeature(task,a_feature,v_feature):
	bestfeature = []
	vote = []
	if task == 'arousal':
		for j in sorted(a_feature, key=lambda i: len(a_feature[i]), reverse=True):
			bestfeature.append(j)
			vote.append(len(a_feature[j]))
			if len(bestfeature)==20:
				break
		print('arousal feature: ',bestfeature)
	
	elif task =='valence':
		for j in sorted(v_feature, key=lambda i: len(v_feature[i]), reverse=True):
			bestfeature.append(j)
			vote.append(len(v_feature[j]))
			if len(bestfeature)==20:
				break
		print('valence feature: ',bestfeature)
	return vote, bestfeature

def allfeature_importance(subject,sort_importance,importance,task, feature_name,a_feature,v_feature):
	if task=='arousal':
		for i in range(collectnum):
			find = np.where(importance==sort_importance[i])[0]
			for j in find:
				name = feature_name[j]
				if name not in a_feature:
					a_feature[name]=[subject]
				else:
					a_feature[name].append(subject)
	
	elif task == 'valence':
		for i in range(collectnum):
			find = np.where(importance==sort_importance[i])[0]
			for j in find:
				name = feature_name[j]
				if name not in v_feature:
					v_feature[name]=[subject]
				else:
					v_feature[name].append(subject)	
	return a_feature, v_feature

def train(X_train, Y_train, X_val, Y_val, subject, threshold, task, feature_name, method, a_feature,v_feature):

	#print('traing...')
	train_acc = np.zeros(threshold)
	train_f1 = np.zeros(threshold)
	val_acc = np.zeros(threshold)
	val_f1 = np.zeros(threshold)
	val_roc = np.zeros(threshold)
	feature_size = np.zeros(threshold)

	xgb1result, train_acc[0], train_f1[0], val_acc[0], val_f1[0], val_roc[0] = modelfit(X_train, Y_train, X_val, Y_val)
	feature_size[0] = xgb1result.n_features_
	ranking = xgb1result.ranking_
	sort_ranking = np.sort(np.unique(ranking))
	a_feature, v_feature = allfeature_importance(subject, sort_ranking, ranking, task, feature_name, a_feature, v_feature)

	print("feat_size: %.3f train f1: %.3f val f1: %.3f roc auc: %.3f" % (feature_size[0],train_f1[0], val_f1[0], val_roc[0]))
	
	folder = 'gsr_result/gsr_fusion/rfe/img/'+method+'/'
	if not os.path.exists(folder):
		os.mkdir(folder)
	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.plot(range(1, len(xgb1result.grid_scores_) + 1), xgb1result.grid_scores_)
	plt.savefig(folder+task+str(subject)+'.png')
	plt.close()
	'''
	for thr in range(threshold):
		if thr==0:
			feature_size[thr] = xgb1result.n
			print(task)
			print("feat_size: %.3f train f1: %.3f val f1: %.3f roc auc: %.3f" % (feature_size[thr],train_f1[thr], val_f1[thr], val_roc[thr]))
		else:
			xgb2result, train_acc[thr], train_f1[thr], val_acc[thr], val_f1[thr], val_roc[thr] = modelfit(X_train, Y_train, X_val, Y_val)

			print("feat_size: %.3f train f1: %.3f val f1: %.3f roc auc: %.3f" % (feature_size[thr],train_f1[thr], val_f1[thr], val_roc[thr]))
	'''
	return train_acc, train_f1, val_acc, val_f1, val_roc, feature_size, a_feature, v_feature

def modelfit(X_train, Y_train, X_val, Y_val):

	parameters = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
	#alg = SVC(C=0.01, kernel='rbf',gamma='auto',probability=True)
	model = SVC(kernel='linear',probability=True)
	test_clf = GridSearchCV(model, parameters, n_jobs=-1, cv=5,verbose=0)
	test_clf.fit(X_train, Y_train)
	best_params =  test_clf.best_params_
	model.set_params(C=best_params['C'])

	alg = RFECV(estimator=model,cv=5,step=1,scoring='neg_log_loss',n_jobs=-1,verbose=0)
	alg.fit(X_train, Y_train)

	Ytrain_pred = alg.predict(X_train)
	Ytrain_pred = (Ytrain_pred>0.5)*1
	Yval_pred = alg.predict(X_val)
	Yval_pred = (Yval_pred>0.5)*1
	Ytrain_pred_prob  = alg.predict_proba(X_train)
	Yval_pred_prob = alg.predict_proba(X_val)

	train_acc = accuracy_score(Y_train,Ytrain_pred) 
	train_f1 = f1_score(Y_train,Ytrain_pred,average='macro')
	#train_loss = log_loss(Y_train,Ytrain_pred_prob)
	val_acc = accuracy_score(Y_val,Yval_pred)
	val_f1 = f1_score(Y_val,Yval_pred,average='macro')
	val_roc = roc_auc_score(Y_val,Yval_pred_prob[:,1])
	#val_loss = log_loss(Y_val, Yval_pred_prob)

	return alg, train_acc, train_f1, val_acc, val_f1, val_roc