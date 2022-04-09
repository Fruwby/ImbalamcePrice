import pandas as pd
import pickle
import numpy as np
import datetime as dt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import time
import sys
import tensorflow as tf
#from tensorflow.keras.layers import LeakyReLu, Dense

def cols_to_delete_per_type(type):

    if type == 'SI':
        cols_to_delete = ['NRV', 'Vol_GCC_I', 'GUV', 'GDV', 'Vol_GCC_D','FCT_t_0', 'Unnamed: 2', 'Unnamed: 3','Unnamed: 4','Unnamed: 5','Unnamed: 6','Unnamed: 7','Unnamed: 8','Unnamed: 9']
    elif type == 'FC_SI':
        cols_to_delete = ['SI', 'NRV', 'Vol_GCC_I', 'GUV', 'GDV', 'Vol_GCC_D']
    else:
        print('Error: invalid train type. Please use SI or FC_SI')
        sys.exit()

    return cols_to_delete

def split_df_features_labels_arrays(dataframe, type='SI'):

    df = dataframe.copy(deep=True)

    cols_to_delete = cols_to_delete_per_type(type)
    df.drop(cols_to_delete,axis=1,inplace=True)

    labels = np.array(df['adjusted_imb_price_alpha'])
    features = np.array(df.drop(['adjusted_imb_price_alpha'], axis=1))

    return features,labels

def train_model(features,labels,type='RF'):
    print('hello world')

    if type == 'RF':
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        t = time.time()
        model.fit(features,labels)
        train_time = time.time()-t

    elif type == 'LM':
        model = LinearRegression()
        t = time.time()
        model.fit(features,labels)
        train_time = time.time()-t

    elif type == 'NN':

        init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero',
                     'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']

        tf.random.set_seed(42)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=20,kernel_initializer=init_mode[4], activation='relu'),
            tf.keras.layers.Dense(units=20,kernel_initializer=init_mode[4], activation='relu'),
            #tf.keras.layers.Dense(units=1, activation='relu'),
            #tf.keras.layers.Dense(units=10, activation='relu'),
            #tf.keras.layers.Dense(units=3, activation='elu'),
            tf.keras.layers.Dense(units=1,activation = 'elu')
        ])

        model.compile(optimizer='Adam', loss='mse')
        t = time.time()
        model.fit(features,labels,epochs=10)
        train_time = time.time() - t

    else:
        print('Error: invalid model type. Please use RF,LM or NN')
        sys.exit()

    return model,train_time

def train_set_performance(features,labels,model,scaler_labels=0,type='RF'):

    if scaler_labels == 0:
        train_predictions = model.predict(features)
    else:
        train_predictions_scaled = model.predict(features).reshape(-1,1)
        train_predictions = unscale_data(train_predictions_scaled,scaler_labels)

    avg_abs_error = np.mean(np.abs(train_predictions - labels))

    return avg_abs_error,train_predictions

def store_model(model,type='RF',code=''):

    loc = 'C:\\Users\\u0137781\\OneDrive - KU Leuven\\Imbalance price\\Python scripts\\trained_models\\'
    dir_loc = loc+type
    full_loc = dir_loc+ '\\' + type + '_' + code + '.sav'

    pickle.dump(model, open(full_loc, 'wb'))

def scale_data(data):
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    transformed = scaler.transform(data)
    return transformed, scaler


# Descale data for post-processing analysis
def unscale_data(data, scaler):
    inversed = scaler.inverse_transform(data)  # inverse transform
    return inversed



if __name__ == '__main__':
    x = tf.reduce_sum(tf.random.normal([1000, 1000]))

    train_set_file_loc = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance price//Data//processed_data//train_set.csv'
    df_train = pd.read_csv(train_set_file_loc).drop(['Unnamed: 0', 'Datetime', 'PPOS', 'MIP', 'MDP', 'Alpha'], axis=1)

    features,labels = split_df_features_labels_arrays(df_train,type='SI')
    labels = labels.reshape(-1,1)
    features_scaled, feature_scaler = scale_data(features)
    labels_scaled, labels_scaler = scale_data(labels)

    trained_model = train_model(features_scaled,labels_scaled,'NN')[0]



    avg_abs_error,train_predictions = train_set_performance(features_scaled,labels,trained_model,labels_scaler)

    store_code  = '20220330'
    store_model(trained_model,code=store_code)
