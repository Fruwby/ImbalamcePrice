import pandas as pd
import pickle
import numpy as np
import datetime as dt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#Analysis for only LA 1


#Load test set, prepare features and labels for different cases of imbalance price calculation

test_set_file_loc =  'C://Users//u0137781//OneDrive - KU Leuven//Imbalance price//Data//processed_data//test_set.csv'

df_test = pd.read_csv(test_set_file_loc).drop(['Unnamed: 0', 'Datetime', 'PPOS', 'MIP', 'MDP', 'Alpha'], axis=1)
FC_SI_cols = ['FCT_t_0', 'Unnamed: 2', 'Unnamed: 3','Unnamed: 4','Unnamed: 5','Unnamed: 6','Unnamed: 7','Unnamed: 8','Unnamed: 9']

SI_values = np.array(df_test['SI'])

#Labels/features for training with forecasted SI
df_test_FC_SI = df_test.drop(['NRV', 'SI', 'Vol_GCC_I', 'GUV', 'GDV', 'Vol_GCC_D'], axis=1)
labels_FC_SI = np.array(df_test_FC_SI['adjusted_imb_price_alpha'])
features_FC_SI = np.array(df_test_FC_SI.drop(['adjusted_imb_price_alpha'], axis=1))

df_test_without_FC_SI = df_test.drop(FC_SI_cols, axis=1)

#Labels/features for training with actual SI
df_test_SI = df_test_without_FC_SI.drop(['NRV', 'Vol_GCC_I', 'GUV', 'GDV', 'Vol_GCC_D'], axis=1)
labels_SI = np.array(df_test_SI['adjusted_imb_price_alpha'])
features_SI = np.array(df_test_SI.drop(['adjusted_imb_price_alpha'], axis=1))

#Labels/features for training with actual SI
df_test_NRV = df_test_without_FC_SI.drop(['SI', 'Vol_GCC_I', 'GUV', 'GDV', 'Vol_GCC_D'], axis=1)
labels_NRV = np.array(df_test_NRV['adjusted_imb_price_alpha'])
features_NRV = np.array(df_test_NRV.drop(['adjusted_imb_price_alpha'], axis=1))

#Labels/features for training with actual GUV and GDV
df_test_GAV = df_test_without_FC_SI.drop(['SI', 'NRV', 'Vol_GCC_I', 'Vol_GCC_D'], axis=1)
labels_GAV = np.array(df_test_GAV['adjusted_imb_price_alpha'])
features_GAV = np.array(df_test_GAV.drop(['adjusted_imb_price_alpha'], axis=1))

#Labels/features for training with actual GUV, GDV and IGCC
df_test_GAV_adj = df_test_without_FC_SI.drop(['SI', 'NRV'], axis=1)
#df_test_GAV_adj['GUV_adj'] = df_test_GAV_adj['GUV'] - df_test_GAV_adj['Vol_GCC_I']
#df_test_GAV_adj['GDV_adj'] = df_test_GAV_adj['GDV'] - df_test_GAV_adj['Vol_GCC_D']
#df_test_GAV_adj.drop(['Vol_GCC_I', 'GUV', 'GDV', 'Vol_GCC_D'], axis=1, inplace=True)
labels_GAV_adj = np.array(df_test_GAV_adj['adjusted_imb_price_alpha'])
features_GAV_adj = np.array(df_test_GAV_adj.drop(['adjusted_imb_price_alpha'], axis=1))


#Load trained RFs for different cases, and make predictions

filename = 'rf_FC_SI.sav'
rf_FC_SI = pickle.load(open(filename, 'rb'))
predictions_FC_SI = rf_FC_SI.predict(features_FC_SI)

filename = 'rf_SI.sav'
rf_SI = pickle.load(open(filename, 'rb'))
predictions_SI = rf_SI.predict(features_SI)

filename = 'rf_NRV.sav'
rf_NRV = pickle.load(open(filename, 'rb'))
predictions_NRV = rf_NRV.predict(features_NRV)

filename = 'rf_GAV.sav'
rf_GAV = pickle.load(open(filename, 'rb'))
predictions_GAV = rf_GAV.predict(features_GAV)

filename = 'rf_GAV_adj.sav'
rf_GAV_adj = pickle.load(open(filename, 'rb'))
predictions_GAV_adj = rf_GAV_adj.predict(features_GAV_adj)



#check performance



errors_FC_SI = predictions_FC_SI- labels_FC_SI
abs_errors_FC_SI = abs(errors_FC_SI)
overall_abs_error_FC_SI = round(np.mean(abs_errors_FC_SI),2)

errors_SI = predictions_SI- labels_SI
abs_errors_SI = abs(errors_SI)
overall_abs_error_SI = round(np.mean(abs_errors_SI),2)

errors_NRV = predictions_NRV- labels_NRV
abs_errors_NRV = abs(errors_NRV)
overall_abs_error_NRV = round(np.mean(abs_errors_NRV),2)

errors_GAV = predictions_GAV- labels_GAV
abs_errors_GAV = abs(errors_GAV)
overall_abs_error_GAV = round(np.mean(abs_errors_GAV),2)

errors_GAV_adj = predictions_GAV_adj- labels_GAV_adj
abs_errors_GAV_adj= abs(errors_GAV_adj)
overall_abs_error_GAV_adj = round(np.mean(abs_errors_GAV_adj),2)

#Interval analysis

max_value = 700
min_value = -700
intervals = 14
delta = (max_value - min_value) / intervals

percentage_correct_FC_SI = np.zeros(intervals)
avg_abs_err_interval_FC_SI = np.zeros(intervals)
avg_err_interval_FC_SI = np.zeros(intervals)

percentage_correct_SI = np.zeros(intervals)
avg_abs_err_interval_SI = np.zeros(intervals)
avg_err_interval_SI = np.zeros(intervals)

percentage_correct_NRV = np.zeros(intervals)
avg_abs_err_interval_NRV= np.zeros(intervals)
avg_err_interval_NRV= np.zeros(intervals)

percentage_correct_GAV = np.zeros(intervals)
avg_abs_err_interval_GAV = np.zeros(intervals)
avg_err_interval_GAV = np.zeros(intervals)

percentage_correct_GAV_adj = np.zeros(intervals)
avg_abs_err_interval_GAV_adj = np.zeros(intervals)
avg_err_interval_GAV_adj = np.zeros(intervals)

for i in range(intervals):
    set = (SI_values[:]>max_value -(i+1)*delta)&(SI_values[:]<max_value -i*delta)

    avg_err_interval_FC_SI[i] = errors_FC_SI[set].mean()
    avg_abs_err_interval_FC_SI[i] = abs_errors_FC_SI[set].mean()

    avg_err_interval_SI[i] = errors_SI[set].mean()
    avg_abs_err_interval_SI[i] = abs_errors_SI[set].mean()

    avg_err_interval_NRV[i] = errors_NRV[set].mean()
    avg_abs_err_interval_NRV[i] = abs_errors_NRV[set].mean()

    avg_err_interval_GAV[i] = errors_GAV[set].mean()
    avg_abs_err_interval_GAV[i] = abs_errors_GAV[set].mean()

    avg_err_interval_GAV_adj[i] = errors_GAV_adj[set].mean()
    avg_abs_err_interval_GAV_adj[i] = abs_errors_GAV_adj[set].mean()


    """
    help = np.array(abs_errors)
    help_interval = help[(test_features[:,20]>max_value -(i+1)*delta)&(test_features[:,20]<max_value -i*delta)]
    avg_abs_error[i] = help_interval.mean()
    percentage_correct[i] = help_interval[help_interval == 0].shape[0] / help_interval.shape[0]
"""

all_errors_interval_FC_SI = np.vstack((avg_abs_err_interval_FC_SI,avg_err_interval_FC_SI))
dump_loc_FC_SI = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance price//Data//processed_data//avg_error_interval_FC_SI.csv'
all_errors_interval_FC_SI.tofile(dump_loc_FC_SI,sep=',')

all_errors_interval_SI = np.vstack((avg_abs_err_interval_SI,avg_err_interval_SI))
dump_loc_SI = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance price//Data//processed_data//avg_error_interval_SI.csv'
all_errors_interval_SI.tofile(dump_loc_SI,sep=',')

all_errors_interval_NRV = np.vstack((avg_abs_err_interval_NRV,avg_err_interval_NRV))
dump_loc_NRV = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance price//Data//processed_data//avg_error_interval_NRV.csv'
all_errors_interval_NRV.tofile(dump_loc_NRV,sep=',')

all_errors_interval_GAV = np.vstack((avg_abs_err_interval_GAV,avg_err_interval_GAV))
dump_loc_GAV = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance price//Data//processed_data//avg_error_interval_GAV.csv'
all_errors_interval_GAV.tofile(dump_loc_GAV,sep=',')

all_errors_interval_GAV_adj = np.vstack((avg_abs_err_interval_GAV_adj,avg_err_interval_GAV_adj))
dump_loc_GAV_adj = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance price//Data//processed_data//avg_error_interval_GAV_adj.csv'
all_errors_interval_GAV_adj.tofile(dump_loc_GAV_adj,sep=',')



print('Mean absolute error:', overall_abs_error_FC_SI, '€/MWh')

"""
#Analysis longer LA
test_set_file_LA_loc = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance price//Data//processed_data//test_set_with_LA.csv'

df_test_set_LA = pd.read_csv(test_set_file_LA_loc).drop(['Datetime','PPOS', 'MIP', 'MDP', 'Alpha','NRV', 'SI'],axis=1)

test_labels_LA= np.array(df_test_set_LA['adjusted_imb_price_alpha'])
test_features_LA = df_test_set_LA.drop(['adjusted_imb_price_alpha'],axis=1)
feature_list_LA = list(test_features_LA.columns)
test_features_LA = np.array(test_features_LA)[:,1:]

max_LA = 48
n_qhours = test_features_LA.shape[0]
#n_qhours = 150
n_quantiles = 9
n_cols_MO = 20

feature_list_LA_combi = np.zeros((n_qhours-max_LA,max_LA,n_quantiles+n_cols_MO))

for qh in range(n_qhours-max_LA):
    for la in range(max_LA):
        for SI_quantile in range(n_quantiles):
            feature_list_LA_combi[qh,la,SI_quantile] = test_features_LA[qh,n_quantiles*la + SI_quantile]
            feature_list_LA_combi[qh,la,n_quantiles:] = test_features_LA[qh,-n_cols_MO:]


labels_LA_combi = np.zeros((n_qhours-max_LA,max_LA))
for qh in range(n_qhours-max_LA):
    for la in range(max_LA):
        labels_LA_combi[qh,la] = test_labels_LA[qh+la]

predictions = np.zeros((n_qhours-max_LA,max_LA))
errors = np.zeros((n_qhours-max_LA,max_LA))

for la in range(max_LA):
    predictions[:,la] = rf.predict(feature_list_LA_combi[:,la,:])
    errors[:,la] = predictions[:,la] - labels_LA_combi[:,la]

abs_errors = np.absolute(errors)
mean_abs_errors = np.mean(abs_errors,axis=0)

dump_loc = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance price//Data//processed_data//mean_abs_error_vs_LA_RF.csv'
mean_abs_errors.tofile(dump_loc,sep=',')

x=1
"""