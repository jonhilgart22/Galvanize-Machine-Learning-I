__author__='Jonathan Hilgart'
from  sklearn.ensemble import GradientBoostingRegressor ##stage plat for MSE
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

class Model_Testing_Regression():
	"""Test a number of different machine learning algorithmns on your data"""

	def __init__(self,X,y,number_of_folds,x_labels,y_label):
		"""Initialize the class with the data and train test split"""

		self.X=X
		self.y=y
		self.number_of_folds=number_of_folds
		self.X_trainval, self.X_test, self.y_trainval, self.y_test = train_test_split(X,y,test_size=.2) ##smaller test size due to less data
		self.x_labels =x_labels
		self.y_label = y_label

	def random_forest(self, number_estimators,m_features, m_depth):
		"""Given the parameters, return the RMSE, accuracy, and feature importance.
		Return str(rmse val), rmse val, str(rmse train),rmse train, str(rmse test), rmse test, feature importance and weight"""

		# Parameters (n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, 
		#min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
		#min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)
		rf_model = RandomForestRegressor(n_estimators=number_estimators,max_depth=m_depth,max_features=m_features)
		rmse_val = []
		rmse_train = []
		rmse_test = []
		feature_importances = []
		feature_importances_final =[]
		for i in range(self.number_of_folds): ## cross validation of the model
			X_train, X_val, y_train, y_val = train_test_split(self.X_trainval,self.y_trainval,test_size=.2)
			rf_model.fit(X_train,y_train)
			rmse_val.append(np.linalg.norm(y_val - rf_model.predict(X_val))/sqrt(len(y_val)))
			rmse_train.append(np.linalg.norm(y_train - rf_model.predict(X_train))/sqrt(len(y_train)))
			rmse_test.append(np.linalg.norm(self.y_test - rf_model.predict(self.X_test))/sqrt(len(self.y_test)))
			feature_importances.append(rf_model.feature_importances_)
		##average the feature importances across the folds
		for element in range(len(rf_model.feature_importances_)):
			list_of_current_feature_numbers = []
			for feature_list in feature_importances:
				list_of_current_feature_numbers.append(feature_list[element])
			feature_importances_final.append(np.mean(list_of_current_feature_numbers))
		feature_importances_final = np.array(feature_importances_final) ## for sorting 
		#sort the features
		sorted_features = self.x_labels[np.argsort(feature_importances_final)[::-1]]
		return 'RMSE Val:',np.mean(rmse_val),'RMSE Train:',np.mean(rmse_train),'RMSE TEST:',np.mean(rmse_test), [(feature,weight)\
		 for feature,weight in zip(sorted_features,feature_importances_final[np.argsort(feature_importances_final)[::-1]])]
	def gradient_boost(self,loss_type=None,learning_rate_n=None,n_estimators_n=None,max_depth_n=None):
		"""Perform gradient boosting given the parameters. 
		Return str(rmse val), rmse val, str(rmse train),rmse train, str(rmse test), rmse test, feature importance and weight """
		########Attributes############
		#loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, 
		#criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, 
		#min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07,
		# init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

		rmse_val = []
		rmse_train = []
		rmse_test = []
		feature_importances = []
		feature_importances_final = []
		for i in range(self.number_of_folds): ## cross validation of the model
			X_train, X_val, y_train, y_val = train_test_split(self.X_trainval,self.y_trainval,test_size=.2)
			gdb_model = GradientBoostingRegressor(loss=loss_type,learning_rate=learning_rate_n,n_estimators=n_estimators_n,max_depth=max_depth_n)
			gdb_model.fit(X_train,y_train) ## fit the model
			rmse_val.append(np.linalg.norm(y_val - gdb_model.predict(X_val))/sqrt(len(y_val)))
			rmse_train.append(np.linalg.norm(y_train - gdb_model.predict(X_train))/sqrt(len(y_train)))
			rmse_test.append(np.linalg.norm(self.y_test - gdb_model.predict(self.X_test))/sqrt(len(self.y_test)))
			feature_importances.append(gdb_model.feature_importances_)
		##average the feature importances across the folds
		for element in range(len(gdb_model.feature_importances_)):
			list_of_current_feature_numbers = []
			for feature_list in feature_importances:
				list_of_current_feature_numbers.append(feature_list[element])
			feature_importances_final.append(np.mean(list_of_current_feature_numbers))
		feature_importances_final = np.array(feature_importances_final) ## for sorting 
		#sort the features
		sorted_features = self.x_labels[np.argsort(feature_importances_final)[::-1]]
		return 'RMSE Val:',np.mean(rmse_val),'RMSE Train:',np.mean(rmse_train),'RMSE TEST:',np.mean(rmse_test), [(feature,weight)\
		 for feature,weight in zip(sorted_features,feature_importances_final[np.argsort(feature_importances_final)[::-1]])]


