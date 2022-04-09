import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import pickle
import numpy as np
import scipy.optimize as so
import copy
import time

df_all_year = pd.read_pickle("df_all_year.pkl")

array_all_year = df_all_year.to_numpy()

filename = 'rf.sav'
rf = pickle.load(open(filename, 'rb'))


n_scenarios = 5
n_qhs = 10

scenarios_imported = np.loadtxt(open('scenarios_qh_1.csv','rb'),delimiter=',')


def return_avg_price_per_qh(MO_and_alpha,SI):
    all_prices = np.zeros((len(SI[:,0]),len(SI[0,:])))
    for scen in range(len(SI[0,:])):
        input = np.concatenate((MO_and_alpha[:, 0:20], SI[:,scen:scen+1], MO_and_alpha[:, 21:22]), axis=1)
        all_prices[:,scen] = rf.predict(input)

    avg_prices = np.mean(all_prices,axis=1)
    return avg_prices


max_charge = 0.5
max_discharge = 0.5
eff_d = 0.9
eff_c = 0.9
max_soc = 2
min_soc = 0
soc_0 = 1
ts_length = 0.25

def objective(x):
    MO_and_SI = copy.deepcopy(array_all_year[0:n_qhs, 0:22])
    MO_and_SI[0:n_qhs, 20] += x[0:n_qhs] - x[n_qhs:2 * n_qhs]
    price = rf.predict(MO_and_SI)
    return(-np.dot(x[0:n_qhs] - x[n_qhs:2 * n_qhs], price))

def objective_scenarios(x):
    n_qhs = len(scenarios_SI[:,0])
    net_discharge_ene = x[0:n_qhs]*eff_d - x[n_qhs:2 * n_qhs]/eff_c
    net_discharge_power = net_discharge_ene/ts_length
    adjusted_SI = np.transpose(scenarios_SI) + x[0:n_qhs] - x[n_qhs:2 * n_qhs]
    avg_prices = return_avg_price_per_qh(array_all_year[0:n_qhs,0:22],np.transpose(adjusted_SI))
    return(-np.dot(net_discharge_ene, avg_prices))



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

"""
def ineq_constr1(x):
    return -x[0] + 100

def ineq_constr2(x):
    return x[0] + 100

def eq_constr(x):
    MO_and_SI = copy.deepcopy(array_all_year[0:1,0:22])
    MO_and_SI[0,20] += x[0]
    return x[1] - rf.predict(MO_and_SI)

constraint1 = {'type': 'ineq', 'fun': ineq_constr1}
constraint2 = {'type': 'ineq', 'fun': ineq_constr2}
constraint3 = {'type': 'eq', 'fun': eq_constr}
constraints = [constraint1,constraint2,constraint3]
"""
ns_SI = [1]
qhs = [1,5,20,40]
times = np.zeros((len(ns_SI),len(qhs)))
for scen in range(len(ns_SI)):
    for la in range(len(qhs)):
        n_scenarios = ns_SI[scen]
        n_qhs = qhs[la]

        #scenarios_SI = (np.random.random((n_qhs, n_scenarios)) - 1 / 2) * 1000
        scenarios_SI = np.transpose(scenarios_imported[0:n_scenarios,0:n_qhs])
        guess_1 = np.concatenate((np.zeros(n_qhs) + 1, np.zeros(n_qhs) - 1, np.ones(n_qhs)), axis=0)

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
        res_1 = so.minimize(objective_scenarios,guess_1,method="SLSQP", tol = 0.001, bounds=bounds,constraints = constraints)
        #res_1 = so.minimize(objective_scenarios,guess_1,method="SLSQP", tol = 0.001, bounds=bounds)
        toc = time.perf_counter()

        times[scen,la] = toc-tic

        filestring = 'opti_variables'+str(ns_SI[scen])+'_'+str(qhs[la])+'.csv'
        variables = res_1['x']
        variables.tofile(filestring, sep=',')


x=1

