import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import pickle
import numpy as np
import scipy.optimize as so
import copy
import time

df_all_year_check = pd.read_pickle("df_all_year_FC_SI.pkl").drop(['Datetime'],axis=1)
array_all_year_check = df_all_year_check.to_numpy()

df_all_year_cols = pd.read_pickle("df_all_year_with_LA.pkl").drop(['Datetime'],axis=1)
array_all_year_cols = df_all_year_cols.to_numpy()

n_merit_order_cols = 20
n_quantiles = 9
n_cols_quantiles = (array_all_year_cols.shape[1] - n_merit_order_cols-2)
range_non_quantile_cols = range(n_cols_quantiles,array_all_year_cols.shape[1])
check = array_all_year_cols[:,range_non_quantile_cols]
max_lookahead = int(n_cols_quantiles/n_quantiles)

array_all_year = np.zeros((array_all_year_cols.shape[0],max_lookahead,n_quantiles+n_merit_order_cols+2))

optimized_qhs = 96*31

for qh in range(optimized_qhs):
    for la in range(max_lookahead):
        range_la = range(n_quantiles*la,n_quantiles*(la+1))
        array_all_year[qh,la,0:n_quantiles] = array_all_year_cols[qh,range_la]
        array_all_year[qh,la,n_quantiles:n_quantiles+n_merit_order_cols+2] = array_all_year_cols[qh+la,range_non_quantile_cols]


cols_SI_FC = range(9)


filename = 'rf_FC_SI.sav'
rf = pickle.load(open(filename, 'rb'))

n_qhs = 10




max_charge = 0.5
max_discharge = 0.5
eff_d = 0.9
eff_c = 0.9
max_soc = 2
min_soc = 0
soc_0 = 1
ts_length = 0.25



def objective(x):
    MO_and_SI = copy.deepcopy(array_all_year[0:n_qhs, 0:30])
    MO_and_SI[0:n_qhs, cols_SI_FC] = np.transpose(np.transpose(MO_and_SI[0:n_qhs, cols_SI_FC]) + np.transpose(x[0:n_qhs] - x[n_qhs:2 * n_qhs]))
    price = rf.predict(MO_and_SI)
    return(-np.dot(x[0:n_qhs] - x[n_qhs:2 * n_qhs], price))


def cons_soc_init(x):
    return x[2 * n_qhs] - soc_0 + x[0] - x[n_qhs]

def cons_soc_update(x):
    net_discharge = -x[1:n_qhs] + x[n_qhs + 1:2 * n_qhs]
    shifted_soc = x[2 * n_qhs:3 * n_qhs - 1]
    soc = x[2 * n_qhs + 1:3 * n_qhs]
    return net_discharge + shifted_soc - soc

constraint1 = {'type': 'eq', 'fun': cons_soc_init}
constraint2 = {'type': 'eq', 'fun': cons_soc_update}
constraints = [constraint1,constraint2]



n_qhs = 5

guess_1 = np.concatenate((np.zeros(n_qhs), np.zeros(n_qhs), np.ones(n_qhs)), axis=0)

lb_dis = np.zeros(n_qhs)
lb_ch = np.zeros(n_qhs)
lb_soc = np.ones(n_qhs) * min_soc
ub_dis = np.ones(n_qhs) * max_discharge
ub_ch = np.ones(n_qhs) * max_charge
ub_soc = np.ones(n_qhs) * max_soc

lb = np.concatenate((lb_dis, lb_ch, lb_soc), axis=0)
ub = np.concatenate((ub_dis, ub_ch, ub_soc), axis=0)

bounds = so.Bounds(lb, ub)

tic = time.perf_counter()
res_1 = so.minimize(objective, guess_1, method="SLSQP", tol=0.001, bounds=bounds, constraints=constraints)
net_discharge,soc_next = res_1['x'][0] - res_1['x'][n_qhs], res_1['x'][2*n_qhs]

qhs = [1,5,20,40]
times = np.zeros(len(qhs))
for la in range(len(qhs)):
    n_qhs = qhs[la]

    guess_1 = np.concatenate((np.zeros(n_qhs), np.zeros(n_qhs), np.ones(n_qhs)), axis=0)

    lb_dis = np.zeros(n_qhs)
    lb_ch = np.zeros(n_qhs)
    lb_soc = np.ones(n_qhs) * min_soc
    ub_dis = np.ones(n_qhs) * max_discharge
    ub_ch = np.ones(n_qhs) * max_charge
    ub_soc = np.ones(n_qhs) * max_soc

    lb = np.concatenate((lb_dis, lb_ch, lb_soc), axis=0)
    ub = np.concatenate((ub_dis, ub_ch, ub_soc), axis=0)

    bounds = so.Bounds(lb, ub)


    tic = time.perf_counter()
    res_1 = so.minimize(objective,guess_1,method="SLSQP", tol = 0.001, bounds=bounds,constraints = constraints)
    #res_1 = so.minimize(objective_scenarios,guess_1,method="SLSQP", tol = 0.001, bounds=bounds)
    toc = time.perf_counter()

    times[la] = toc-tic

    filestring = 'opti_variables_FC_SI_'+str(qhs[la])+'.csv'
    variables = res_1['x']
    variables.tofile(filestring, sep=',')


x=1

