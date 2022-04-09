import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import pickle
import numpy as np
import scipy.optimize as so
import copy
import time

#Function to convert input 2d array to usable 3d array for optimization loop
def convert_columns_array_to_split_array(array,n_quantiles,n_merit_order_cols,n_other_cols,optimized_qhs):
    n_cols_quantiles = (array.shape[1] - n_merit_order_cols - n_other_cols)
    range_non_quantile_cols = range(n_cols_quantiles, array.shape[1])
    max_lookahead = int(n_cols_quantiles / n_quantiles)

    array_all_year = np.zeros((array_all_year_cols.shape[0], max_lookahead, n_quantiles + n_merit_order_cols + n_other_cols))

    for qh in range(optimized_qhs):
        for la in range(max_lookahead):
            range_la = range(n_quantiles * la, n_quantiles * (la + 1))
            array_all_year[qh, la, 0:n_quantiles] = array[qh, range_la]
            array_all_year[qh, la, n_quantiles:n_quantiles + n_merit_order_cols + n_other_cols] = array[qh + la, range_non_quantile_cols]

    return array_all_year


#Load data and convert to 2d array
df_all_year_cols = pd.read_pickle("df_all_year_with_LA.pkl").drop(['Datetime'],axis=1)
array_all_year_cols = df_all_year_cols.to_numpy()

#Define parameters required as input for conversion function
n_merit_order_cols = 20
n_quantiles = 9
n_other_cols = 2
n_qhs = 96

#Convert input array to 3d array
array_all_year = convert_columns_array_to_split_array(array_all_year_cols,n_quantiles,n_merit_order_cols,n_other_cols,n_qhs)

#Load trained random forest
filename = 'rf_FC_SI.sav'
rf = pickle.load(open(filename, 'rb'))

#Define ESS parameters
max_charge = 0.01
max_discharge = 0.01
eff_d = 0.9
eff_c = 0.9
max_soc = 0.04
min_soc = 0
soc_0 = 0.02
ts_length = 0.25
lookahead = 10

#Define arrays to capture optimization output in the loop
optimized_array = np.zeros((n_qhs,3*lookahead))
forward_schedule = np.zeros((n_qhs,lookahead))
forward_soc = np.zeros((n_qhs,lookahead))
expected_price = np.zeros((n_qhs,lookahead))
optimization_times = np.zeros(n_qhs)

for qh in range(n_qhs):
    for la in range(lookahead):
        expected_price[qh,la] = rf.predict(array_all_year[qh,la:la+1, 0:30])


#Define upper and lower bounds for optimization, which don't change throughout the loop
lb_dis = np.zeros(lookahead)
lb_ch = np.zeros(lookahead)
lb_soc = np.ones(lookahead) * min_soc
ub_dis = np.ones(lookahead) * max_discharge
ub_ch = np.ones(lookahead) * max_charge
ub_soc = np.ones(lookahead) * max_soc

lb = np.concatenate((lb_dis, lb_ch, lb_soc), axis=0)
ub = np.concatenate((ub_dis, ub_ch, ub_soc), axis=0)
bounds = so.Bounds(lb, ub)


for qh in range(n_qhs):

    if qh > 0:
        soc_0 = forward_soc[qh-1,0]


    def objective(x):
        MO_and_SI = copy.deepcopy(array_all_year[qh,0:lookahead, 0:30])
        MO_and_SI[:, n_quantiles] = np.transpose(np.transpose(MO_and_SI[:, n_quantiles]) + np.transpose(x[0:lookahead] - x[lookahead:2 * lookahead]))
        price = rf.predict(MO_and_SI)
        return(-np.dot(x[0:lookahead] - x[lookahead:2 * lookahead], price))

    def cons_soc_init(x):
        return x[2 * lookahead] - soc_0 + x[0] - x[lookahead]

    def cons_soc_update(x):
        net_discharge = -x[1:lookahead] + x[lookahead + 1:2 * lookahead]
        shifted_soc = x[2 * lookahead:3 * lookahead - 1]
        soc = x[2 * lookahead + 1:3 * lookahead]
        return net_discharge + shifted_soc - soc

    constraint1 = {'type': 'eq', 'fun': cons_soc_init}
    constraint2 = {'type': 'eq', 'fun': cons_soc_update}
    constraints = [constraint1,constraint2]


    guess_1 = np.concatenate((np.zeros(lookahead), np.zeros(lookahead), np.ones(lookahead)), axis=0)





    tic = time.perf_counter()
    res_1 = so.minimize(objective, guess_1, method="SLSQP", tol=0.001, bounds=bounds, constraints=constraints)
    toc = time.perf_counter()

    optimized_array[qh,:] = res_1['x']
    forward_schedule[qh,:] = res_1['x'][0:lookahead] - res_1['x'][lookahead:2*lookahead]
    forward_soc[qh,:] = res_1['x'][2*lookahead:3*lookahead]
    optimization_times[qh] = toc-tic


filestring = 'forward_schedule_opti_1day_rf.csv'
np.savetxt(filestring,forward_schedule,delimiter=',')


"""

variables = forward_schedule
variables.tofile(filestring, sep=',')
"""
x=1

