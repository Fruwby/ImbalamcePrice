import pandas as pd
import pickle
import numpy as np
import datetime as dt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#Analysis for only LA 1

test_set_file_loc =  'C://Users//u0137781//OneDrive - KU Leuven//Imbalance price//Data//processed_data//test_set.csv'
test_set_file_LA_loc = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance price//Data//processed_data//test_set_with_LA.csv'

df_test_set = pd.read_csv(test_set_file_loc).drop(['Datetime','PPOS', 'MIP', 'MDP', 'Alpha'],axis=1)
df_test_set_LA = pd.read_csv(test_set_file_LA_loc).drop(['Datetime','PPOS', 'MIP', 'MDP', 'Alpha'],axis=1)



test_labels = np.array(df_test_set['adjusted_imb_price_alpha'])
test_features = df_test_set.drop(['adjusted_imb_price_alpha'],axis=1)
feature_list = list(test_features.columns)
test_features = np.array(test_features)[:,1:]

filename = 'rf_FC_SI.sav'
rf = pickle.load(open(filename, 'rb'))

predictions = rf.predict(test_features)

errors = predictions-test_labels
abs_errors = abs(errors)

overall_abs_error = round(np.mean(abs_errors),2)

print('Mean absolute error:', overall_abs_error, '€/MWh')


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


max_value = 700
min_value = -700
intervals = 14
delta = (max_value - min_value) / intervals

percentage_correct = np.zeros(intervals)
avg_abs_error = np.zeros(intervals)
avg_error = np.zeros(intervals)

for i in range(intervals):
    help = np.array(errors)
    help_interval = help[(test_features[:,20]>max_value -(i+1)*delta)&(test_features[:,20]<max_value -i*delta)]
    avg_error[i] = help_interval.mean()

    help = np.array(abs_errors)
    help_interval = help[(test_features[:,20]>max_value -(i+1)*delta)&(test_features[:,20]<max_value -i*delta)]
    avg_abs_error[i] = help_interval.mean()
    percentage_correct[i] = help_interval[help_interval == 0].shape[0] / help_interval.shape[0]

print(percentage_correct)

