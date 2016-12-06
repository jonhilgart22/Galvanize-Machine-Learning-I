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
from scipy.spatial.distance import euclidean
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV

class Model_Testing_Regression():
	"""Test a number of different machine learning regression algorithmns on your data. Need to manually optimize, no gridsearch."""

	def __init__(self,X,y,number_of_folds,x_labels,y_label):
		"""Initialize the class with the data and train test split"""

		self.X=X
		self.y=y
		self.number_of_folds=number_of_folds
		self.X_trainval, self.X_test, self.y_trainval, self.y_test = train_test_split(X,y,test_size=.2) ##smaller test size due to less data
		self.x_labels =x_labels
		self.y_label = y_label

	def random_forest(self, number_estimators=10,n_features='auto', m_depth=None):
		"""Given the parameters, return the RMSE, accuracy, and feature importance.
		Return str(rmse val), rmse val, str(rmse train),rmse train, str(rmse test), rmse test, feature importance and weight"""
		#############attributes#################
		# Parameters (n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, 
		#min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
		#min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)
		rf_model = RandomForestRegressor(n_estimators=number_estimators,max_depth=m_depth,max_features=n_features)
		rmse_val = []
		rmse_train = []
		rmse_test = []
		feature_importances = []
		feature_importances_final =[]
		for i in range(self.number_of_folds): ## cross validation of the model
			X_train, X_val, y_train, y_val = train_test_split(self.X_trainval,self.y_trainval,test_size=.2)
			rf_model.fit(X_train,y_train)
			rmse_val.append(euclidean(y_val,rf_model.predict(X_val))/sqrt(len(y_val)))
			rmse_train.append(euclidean(y_train,rf_model.predict(X_train))/sqrt(len(y_train)))
			rmse_test.append(euclidean(self.y_test,rf_model.predict(self.X_test))/sqrt(len(self.y_test)))
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
	def gradient_boost(self,loss_type='ls',learning_rate_n=.1,n_estimators_n=100,max_depth_n=3):
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
			rmse_val.append(euclidean(y_val,gdb_model.predict(X_val))/sqrt(len(y_val)))
			rmse_train.append(euclidean(y_train,gdb_model.predict(X_train))/sqrt(len(y_train)))
			rmse_test.append(euclidean(self.y_test,gdb_model.predict(self.X_test))/sqrt(len(self.y_test)))
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

	def extreme_gradient_boost(self,max_depth_n=3,learning_rate_n=.1,n_estimators_n=100,reg_alpha_n=0,reg_lambda_n=0):
		"""Perform extreme gradient boosting on the given data. 
		Returns str(rmse val), rmse val, str(rmse train),rmse train, str(rmse test), rmse test, feature importance and weight """


		#######Attributes###########
		#xgboost.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True,
		 #objective='reg:linear', nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, 
		 #subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, 
		 #scale_pos_weight=1, base_score=0.5, seed=0, missing=None)
		rmse_val = []
		rmse_train = []
		rmse_test = []
		feature_importances = []
		feature_importances_final = []
		for i in range(self.number_of_folds): ## cross validation of the model
			X_train, X_val, y_train, y_val = train_test_split(self.X_trainval,self.y_trainval,test_size=.2)
			xgb_model = xgboost.XGBRegressor(max_depth=max_depth_n,learning_rate=learning_rate_n,n_estimators=n_estimators_n,reg_alpha=reg_alpha_n,reg_lambda=reg_lambda_n)
			xgb_model.fit(X_train,y_train) ## fit the model
			rmse_val.append(euclidean(y_val,xgb_model.predict(X_val))/sqrt(len(y_val)))
			rmse_train.append(euclidean(y_train,xgb_model.predict(X_train))/sqrt(len(y_train)))
			rmse_test.append(euclidean(self.y_test,xgb_model.predict(self.X_test))/sqrt(len(self.y_test)))
			feature_importances.append(xgb_model.feature_importances_)
		##average the feature importances across the folds
		for element in range(len(xgb_model.feature_importances_)):
			list_of_current_feature_numbers = []
			for feature_list in feature_importances:
				list_of_current_feature_numbers.append(feature_list[element])
			feature_importances_final.append(np.mean(list_of_current_feature_numbers))
		feature_importances_final = np.array(feature_importances_final) ## for sorting 
		#sort the features
		sorted_features = self.x_labels[np.argsort(feature_importances_final)[::-1]]
		return 'RMSE Val:',np.mean(rmse_val),'RMSE Train:',np.mean(rmse_train),'RMSE TEST:',np.mean(rmse_test), [(feature,weight)\
		 for feature,weight in zip(sorted_features,feature_importances_final[np.argsort(feature_importances_final)[::-1]])]


	def glm_net(self,alpha_n=1.0,l1_ratio_n=.5,normalize_f=False):
		"""Use GLM net with the given parameters to produce a regression model.
		Returns str(rmse val), rmse val, str(rmse train),rmse train, str(rmse test), rmse test, feature importance and weight """
		########Attributes########
		#ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, 
		#	precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
		
		rmse_val = []
		rmse_train = []
		rmse_test = []
		feature_importances = []
		feature_importances_final = []
		for i in range(self.number_of_folds): ## cross validation of the model
			X_train, X_val, y_train, y_val = train_test_split(self.X_trainval,self.y_trainval,test_size=.2)
			ENet_model =  ElasticNet(alpha=alpha_n,l1_ratio=l1_ratio_n,normalize=normalize_f) ## no need to normalize - all columns are in dollars
			ENet_model.fit(X_train,y_train) ## fit the model
			rmse_val.append(euclidean(y_val,ENet_model.predict(X_val))/sqrt(len(y_val)))
			rmse_train.append(euclidean(y_train,ENet_model.predict(X_train))/sqrt(len(y_train)))
			rmse_test.append(euclidean(self.y_test,ENet_model.predict(self.X_test))/sqrt(len(self.y_test)))
			feature_importances.append(abs(ENet_model.coef_)) ## take absolute value to rank coefficients
		##average the feature importances across the folds
		for element in range(len(ENet_model.coef_)):
			list_of_current_feature_numbers = []
			for feature_list in feature_importances:
				list_of_current_feature_numbers.append(feature_list[element])
			feature_importances_final.append(np.mean(list_of_current_feature_numbers))
		feature_importances_final = np.array(feature_importances_final) ## for sorting 
		#sort the features
		sorted_features = self.x_labels[np.argsort(feature_importances_final)[::-1]]
		return 'RMSE Val:',np.mean(rmse_val),'RMSE Train:',np.mean(rmse_train),'RMSE TEST:',np.mean(rmse_test), [(feature,weight)\
		 for feature,weight in zip(sorted_features,feature_importances_final[np.argsort(feature_importances_final)[::-1]])]

	def random_grid_search(self,model=None,params_dict=None,iterations=100,cv_n=3):
		"""params is a dict of the parameters for the model. Return the best model parameters. Model is text for the model selection."""

		if model = 'glm_net':

			random_search = RandomizedSearchCV(estimator=self.glm_net(),param_distributions=params_dict,cv=cv_n,n_iter=iterations,verbose=1)
			random_search.fit(self.X_trainval,self,y_trainval)
			return random_search.best_estimator_
		elif model = 'extreme_gradient_boost':

			random_search = RandomizedSearchCV(estimator=self.extreme_gradient_boost(),param_distributions=params_dict,cv=cv_n,n_iter=iterations,verbose=1)
			random_search.fit(self.X_trainval,self,y_trainval)
			return random_search.best_estimator_

		elif model = 'gradient_boost':

			random_search = RandomizedSearchCV(estimator=self.gradient_boost(),param_distributions=params_dict,cv=cv_n,n_iter=iterations,verbose=1)
			random_search.fit(self.X_trainval,self,y_trainval)
			return random_search.best_estimator_

		elif model = 'random_forest':

			random_search = RandomizedSearchCV(estimator=self.random_forest(),param_distributions=params_dict,cv=cv_n,n_iter=iterations,verbose=1)
			random_search.fit(self.X_trainval,self,y_trainval)
			return random_search.best_estimator_











