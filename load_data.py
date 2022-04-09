import pandas as pd
#test
import numpy as np
import datetime as dt
from datetime import date
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
months = [1,2,3,4,5,6,7,8,9,10,11,12]
years = [2019,2020,2021]


folder_ARC_data = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance price//Data//2020_meritOrder'
folder_SI_imbPrice_data = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance price//Data//2020_SI_NRV_ImbPrice'
folder_activated_volumes = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance price//Data//2020_activated_volumes'
folder_SI_FC = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance price//Data//2020_forecast_SI'

datetime_format = "%d/%m/%Y %H:%M"

def string_from_month(month):
    if month < 10:
        return '0' + str(month)
    else:
        return str(month)

#Load ARC, actual SI, NRV, alpha data per month and merge
for month in months:
    file_loc_ARC = folder_ARC_data + '//ARC_VolumeLevelPrices_2020_'+string_from_month(month)+'.csv'
    file_loc_SI_imbPrice = folder_SI_imbPrice_data + '//ImbalanceNrvPrices_2020'+string_from_month(month)+'.csv'
    file_loc_act_vol = folder_activated_volumes + '//ActivatedEnergyVolumes_2020'+string_from_month(month)+'.csv'

    df_ARC = pd.read_csv(file_loc_ARC, delimiter=';')
    df_ARC = df_ARC.rename({'#NAME?': '-Max'}, axis=1)
    columns = list(df_ARC.columns)

    for i in range(2,12):
        df_ARC[columns[i]].fillna(df_ARC['-Max'], inplace=True)

    for i in range(12,22):
        df_ARC[columns[i]].fillna(df_ARC['Max'], inplace=True)

    df_ARC['Datetime'] = pd.to_datetime(df_ARC['Quarter'].str[:16], format=datetime_format)
    df_ARC.drop(['Quarter', '-Max', 'Max'], axis=1,inplace=True)

    df_SI_imbPrice = pd.read_csv(file_loc_SI_imbPrice,decimal=',')[['NRV','SI','Alpha', 'PPOS', 'EXECDATE', 'QUARTERHOUR','MIP', 'MDP']]
    df_SI_imbPrice['Datetime_string'] = df_SI_imbPrice['EXECDATE'] + ' ' + df_SI_imbPrice['QUARTERHOUR'].str[:5]
    df_SI_imbPrice['Datetime'] = pd.to_datetime(df_SI_imbPrice['Datetime_string'], format=datetime_format)
    df_SI_imbPrice.drop(['EXECDATE','QUARTERHOUR','Datetime_string'],axis=1,inplace=True)

    df_act_vol = pd.read_csv(file_loc_act_vol, delimiter=',', decimal=',')[['execdate', 'strQuarter', 'GUV', 'Vol_GCC_I', 'GDV', 'Vol_GCC_D']]
    df_act_vol['Datetime_string'] = df_act_vol['execdate'] + ' ' + df_act_vol['strQuarter'].str[:5]
    df_act_vol['Datetime'] = pd.to_datetime(df_act_vol['Datetime_string'], format=datetime_format)
    df_act_vol.drop(['execdate', 'strQuarter', 'Datetime_string'], axis=1,inplace=True)
    df_act_vol.replace(np.nan, 0,inplace=True)

    df_combined_month = pd.merge(df_ARC,df_SI_imbPrice,on='Datetime')
    df_combined_month = pd.merge(df_combined_month,df_act_vol,on='Datetime')

    if month == 1:
        df_all_year = df_combined_month
    else:
        df_all_year = pd.concat([df_all_year,df_combined_month])

#Calculate imbalance price without alpha

#check if imbalance price calculation using alpha and MIP/MDP is correct
"""
df_all_year['imb_price_calc'] = np.where(df_all_year['SI']>0,df_all_year['MDP'] - df_all_year['Alpha'], df_all_year['MIP'] + df_all_year['Alpha'])
df_all_year['Check_calculated_correctly'] = abs(df_all_year['imb_price_calc'] - df_all_year['PPOS'])<0.02
df_check = df_all_year[df_all_year['Check_calculated_correctly'] == False]
"""
df_all_year['adjusted_imb_price_alpha'] = np.where(df_all_year['SI']>0,df_all_year['MDP'], df_all_year['MIP'])

#Get forecasted SI
file_loc_FC = folder_SI_FC + '//y_hat_probabilistic_test_combi.csv'
df_FC_SI = pd.read_csv(file_loc_FC,delimiter=';')

df_FC_SI['Datetime'] = pd.to_datetime(df_FC_SI['Unnamed: 0'],format=datetime_format)
df_FC_SI['Month'] = pd.DatetimeIndex(df_FC_SI['Datetime']).month

columns_to_drop = ['Unnamed: 0','Month']
columns_to_keep = ['Datetime', 'FCT_t_0', 'Unnamed: 2', 'Unnamed: 3','Unnamed: 4','Unnamed: 5','Unnamed: 6','Unnamed: 7','Unnamed: 8','Unnamed: 9']
filtered_df_FC_SI = df_FC_SI[columns_to_keep]
filtered_df_FC_SI_with_LA = df_FC_SI.drop(columns_to_drop,axis=1)
col = filtered_df_FC_SI_with_LA.pop('Datetime')
filtered_df_FC_SI_with_LA.insert(0,'Datetime', col)


#Merge data in large dataframes

df_merged_FC_SI = pd.merge(filtered_df_FC_SI,df_all_year,on='Datetime')
df_merged_with_LA = pd.merge(filtered_df_FC_SI_with_LA,df_all_year,on='Datetime')

#Remove days of time changes; 29/03/2020 and 25/10/2020
#Additionally remove 31/12/2020 and 01/01/2020
date_march = dt.date(2020,3,29)
date_october = dt.date(2020,10,25)
date_december = dt.date(2020,12,30) ## why does it say 30th of December here??
date_january = dt.date(2020,1,1)
dates_to_delete = [date_march, date_october, date_december, date_january]

df_merged_FC_SI_filtered = df_merged_FC_SI[~df_merged_FC_SI['Datetime'].dt.date.isin(dates_to_delete)]
df_merged_with_LA_filtered = df_merged_with_LA[~df_merged_with_LA['Datetime'].dt.date.isin(dates_to_delete)]

#Check if GUV/GDV correspond to NRV, and that IGCC is <= GUV or GDV
def NRV_GUV_GDV_check(df_merged_FC_SI_filtered):

    df_merged_FC_SI_filtered['NRV_check'] = abs(df_merged_FC_SI_filtered['GUV'] - df_merged_FC_SI_filtered['GDV'] - df_merged_FC_SI_filtered['NRV']) <0.1
    NRV_check_counter = len(df_merged_FC_SI_filtered) - df_merged_FC_SI_filtered['NRV_check'].sum()
    df_merged_FC_SI_filtered['GUV_check'] = df_merged_FC_SI_filtered['GUV'] - df_merged_FC_SI_filtered['Vol_GCC_I'] > -0.01
    GUV_check_counter = len(df_merged_FC_SI_filtered) - df_merged_FC_SI_filtered['GUV_check'].sum()
    df_merged_FC_SI_filtered['GDV_check'] = df_merged_FC_SI_filtered['GDV'] - df_merged_FC_SI_filtered['Vol_GCC_D'] > -0.01
    GDV_check_counter = len(df_merged_FC_SI_filtered) - df_merged_FC_SI_filtered['GDV_check'].sum()

    return [NRV_check_counter,GUV_check_counter,GDV_check_counter]

if NRV_GUV_GDV_check(df_merged_FC_SI_filtered) == [0,0,0]:
    print('GUV - GDV - NRV check ok')
    df_merged_FC_SI_filtered.drop(['NRV_check','GUV_check','GDV_check'],axis=1,inplace=True)
    #sys.exit()
else:
    print('GUV - GDV - NRV check NOT ok')
    sys.exit()

#Store in csv files

folder_export_data = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance price//Data//processed_data'

threshold_day_train_test = 20

if df_merged_FC_SI_filtered['Datetime'].is_unique:
    print("You're good")
    df_train_set = df_merged_FC_SI_filtered[(df_merged_FC_SI_filtered['Datetime'].dt.day < threshold_day_train_test)]
    df_test_set = df_merged_FC_SI_filtered[(df_merged_FC_SI_filtered['Datetime'].dt.day >= threshold_day_train_test)]
    df_with_LA_train = df_merged_with_LA_filtered[(df_merged_with_LA_filtered['Datetime'].dt.day < threshold_day_train_test)]
    df_with_LA_test = df_merged_with_LA_filtered[(df_merged_with_LA_filtered['Datetime'].dt.day >= threshold_day_train_test)]

    df_train_set.to_csv(folder_export_data+'//train_set.csv')
    df_test_set.to_csv(folder_export_data + '//test_set.csv')
    df_with_LA_train.to_csv(folder_export_data+'//train_set_with_LA.csv')
    df_with_LA_test.to_csv(folder_export_data + '//test_set_with_LA.csv')

df_merged_FC_SI.to_pickle("df_all_year_FC_SI.pkl")
df_merged_with_LA.to_pickle("df_all_year_with_LA.pkl")

print(df_all_year.describe())




