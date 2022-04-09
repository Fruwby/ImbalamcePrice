import pandas as pd
import pickle
import numpy as np
import datetime as dt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time

#Train with actual SI
"""
df_all_year = pd.read_pickle("df_all_year_actual_SI.pkl")

labels = np.array(df_all_year['PPOS'])
features = df_all_year.drop(['PPOS'],axis=1)
feature_list = list(features.columns)
features = np.array(features)

train_features,test_features,train_labels,test_labels = train_test_split(features,labels,test_size = 0.99, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features,train_labels)

filename = 'rf.sav'
pickle.dump(rf, open(filename, 'wb'))

predictions = rf.predict(test_features)
"""
#Train with forecasted SI

train_set_file_loc = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance price//Data//processed_data//train_set.csv'
df_train = pd.read_csv(train_set_file_loc).drop(['Unnamed: 0','Datetime','PPOS', 'MIP', 'MDP', 'Alpha'],axis=1)

FC_SI_cols = ['FCT_t_0', 'Unnamed: 2', 'Unnamed: 3','Unnamed: 4','Unnamed: 5','Unnamed: 6','Unnamed: 7','Unnamed: 8','Unnamed: 9']

#Labels/features for training with forecasted SI
df_train_FC_SI = df_train.drop(['NRV', 'SI','Vol_GCC_I','GUV', 'GDV', 'Vol_GCC_D'],axis=1)
labels_FC_SI = np.array(df_train_FC_SI['adjusted_imb_price_alpha'])
features_FC_SI = np.array(df_train_FC_SI.drop(['adjusted_imb_price_alpha'],axis=1))

df_train_without_FC_SI = df_train.drop(FC_SI_cols,axis=1)

#Labels/features for training with actual SI
df_train_SI = df_train_without_FC_SI.drop(['NRV','Vol_GCC_I','GUV', 'GDV', 'Vol_GCC_D'],axis=1)
labels_SI = np.array(df_train_SI['adjusted_imb_price_alpha'])
features_SI = np.array(df_train_SI.drop(['adjusted_imb_price_alpha'],axis=1))

#Labels/features for training with actual SI
df_train_NRV = df_train_without_FC_SI.drop(['SI','Vol_GCC_I','GUV', 'GDV', 'Vol_GCC_D'],axis=1)
labels_NRV = np.array(df_train_NRV['adjusted_imb_price_alpha'])
features_NRV = np.array(df_train_NRV.drop(['adjusted_imb_price_alpha'],axis=1))

#Labels/features for training with actual GUV and GDV
df_train_GAV = df_train_without_FC_SI.drop(['SI','NRV','Vol_GCC_I', 'Vol_GCC_D'],axis=1)
labels_GAV = np.array(df_train_GAV['adjusted_imb_price_alpha'])
features_GAV = np.array(df_train_GAV.drop(['adjusted_imb_price_alpha'],axis=1))

#Labels/features for training with actual GUV, GDV and IGCC
df_train_GAV_adj = df_train_without_FC_SI.drop(['SI','NRV'],axis=1)
#df_train_GAV_adj['GUV_adj'] = df_train_GAV_adj['GUV'] - df_train_GAV_adj['Vol_GCC_I']
#df_train_GAV_adj['GDV_adj'] = df_train_GAV_adj['GDV'] - df_train_GAV_adj['Vol_GCC_D']
#df_train_GAV_adj.drop(['Vol_GCC_I','GUV', 'GDV', 'Vol_GCC_D'],axis=1,inplace=True)
labels_GAV_adj = np.array(df_train_GAV_adj['adjusted_imb_price_alpha'])
features_GAV_adj = np.array(df_train_GAV_adj.drop(['adjusted_imb_price_alpha'],axis=1))

feature_names = list(df_train_GAV_adj.drop(['adjusted_imb_price_alpha'],axis=1).columns)

#Make train test split and train
"""
train_features,test_features,train_labels,test_labels = train_test_split(features_SI,labels_SI,test_size = 0.5, random_state = 1)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

rf = RandomForestRegressor(n_estimators=100,random_state = 1)
rf.fit(train_features,train_labels)

predictions = rf.predict(test_features)
avg_abs_error = np.mean(np.abs(predictions-test_labels))
print('Absolute error for SI:', avg_abs_error)

"""
#train RF for forecasted SI
rf_FC_SI = RandomForestRegressor(n_estimators = 20, random_state = 42)
rf_FC_SI.fit(features_FC_SI,labels_FC_SI)

train_predictions_FC_SI = rf_FC_SI.predict(features_FC_SI)
avg_abs_error_FC_SI = np.mean(np.abs(train_predictions_FC_SI-labels_FC_SI))

filename = 'rf_FC_SI.sav'
pickle.dump(rf_FC_SI, open(filename, 'wb'))

#train RF for actual SI
rf_SI = RandomForestRegressor(n_estimators = 20, random_state = 42)
rf_SI.fit(features_SI,labels_SI)

train_predictions_SI = rf_SI.predict(features_SI)
avg_abs_error_SI = np.mean(np.abs(train_predictions_SI-labels_SI))

filename = 'rf_SI.sav'
pickle.dump(rf_SI, open(filename, 'wb'))

#train RF for actual NRV
rf_NRV = RandomForestRegressor(n_estimators = 20, random_state = 42)
rf_NRV.fit(features_NRV,labels_NRV)

train_predictions_NRV = rf_NRV.predict(features_NRV)
avg_abs_error_NRV = np.mean(np.abs(train_predictions_NRV-labels_NRV))

filename = 'rf_NRV.sav'
pickle.dump(rf_NRV, open(filename, 'wb'))

#train RF for actual GUV and GDV
rf_GAV = RandomForestRegressor(n_estimators = 20, random_state = 42)
rf_GAV.fit(features_GAV,labels_GAV)

train_predictions_GAV = rf_GAV.predict(features_GAV)
avg_abs_error_GAV = np.mean(np.abs(train_predictions_GAV-labels_GAV))

filename = 'rf_GAV.sav'
pickle.dump(rf_GAV, open(filename, 'wb'))

#train RF for adjusted GUV and GDV for IGCC
t = time.time()
rf_GAV_adj = RandomForestRegressor(n_estimators = 20, random_state = 42)
rf_GAV_adj.fit(features_GAV_adj,labels_GAV_adj)
time_for_training = time.time()-t

train_predictions_GAV_adj = rf_GAV_adj.predict(features_GAV_adj)
avg_abs_error_GAV_adj = np.mean(np.abs(train_predictions_GAV_adj-labels_GAV_adj))

filename = 'rf_GAV_adj.sav'
pickle.dump(rf_GAV_adj, open(filename, 'wb'))


import sklearn.tree
import matplotlib.pyplot as plt

estimator = rf_GAV_adj.estimators_[0]



fig = plt.figure()
sklearn.tree.plot_tree(estimator,feature_names = feature_names,filled=True,impurity=True,rounded=True)


