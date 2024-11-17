import numpy as np
from pyomo import environ as pe
from pyomo.environ import *
import time
import pandas as pd
from scipy.stats import uniform, randint
from pyomo.opt import SolverFactory
from src.problem.math_solver.KnapsackGenerator import MultiKnapsackGenerator

def knapsack_synergy(num_var, num_ineq, c, raw_p, raw_w):

    # convert lists to dict
    p = {} # prices
    w = {} # weights
    q = {}

    for j in range(num_var):
        p[j] = raw_p[j]

    for i in range(num_ineq):
        for j in range(num_var):
            w[i,j] =raw_w[i][j]

    for j1 in range(num_var):
        for j2 in range(num_var):
            q[j1,j2] = 0.01*(p[j1]+p[j2])

    # Mathematical model
    m = pe.ConcreteModel()
    # fixed parameters
    m.p = pe.Param(pe.RangeSet(0,num_var-1), initialize=p)
    m.w = pe.Param(pe.RangeSet(0,num_ineq-1), pe.RangeSet(0,num_var-1), initialize=w)
    m.q = pe.Param(pe.RangeSet(0,num_var-1), pe.RangeSet(0,num_var-1), initialize=q)
    # mutable parameters (parametric part of the problem)
    m.c = pe.Param(pe.RangeSet(0,num_ineq-1), initialize=c)
    # decision variables
    m.x = pe.Var(pe.RangeSet(0,num_var-1), domain=pe.NonNegativeReals)
    # objective function
    obj = sum([-1*m.x[j] * m.p[j]  for j in range(num_var)]) + sum([-1*m.q[j1,j2]*m.x[j1]*m.x[j2] for j1 in range(num_var-1) for j2 in range(j1+1,num_var)])
    m.obj = pe.Objective(sense=pe.minimize, expr=obj)
    # constraints
    m.cons = pe.ConstraintList()
    for i in range(num_ineq):
        m.cons.add( sum([m.w[i,j]*m.x[j] for j in range(num_var)]) <= m.c[i] )
    
    return m



prob_size_list = [[5, 5] , [10, 10], [20, 20], [50, 50], [100, 100]] # [[num_var, num_ineq]]
num_data = 100


for num_var, num_ineq in prob_size_list:


    print('-'*70)
    print('-'*70)
    print('Initiating for problems of the size of %i * %i' % (num_var, num_ineq))
    print('-'*70)


    objvals, conviols, elapseds = [], [], []


    print('Generating %i number of problem instances'%(num_data))

    tick = time.time()
    samples = MultiKnapsackGenerator(
                n=randint(low=num_var, high=num_var+1),
                m=randint(low=num_ineq, high=num_ineq+1),
                w=uniform(loc=0, scale=60),
                K=uniform(loc=100, scale=0),
                u=uniform(loc=1, scale=0),
                alpha=uniform(loc=0.25, scale=0),
                w_jitter=uniform(loc=0.95, scale=0.1),
                p_jitter=uniform(loc=0.75, scale=0.5),
                rng_state=17
            ).generate(num_data)
    
    tock = time.time()
    

    print('End of generating problem instances (%f sec).' % (tock - tick))
    print('-'*70)


    c_samples = np.array([samples[i].capacities for i in range(num_data)])

    raw_p = samples[0].prices
    raw_w = samples[0].weights


    print('Solving...')
    counter = 0


    for c in c_samples:
        # init model
        model = knapsack_synergy(num_var, num_ineq, c, raw_p, raw_w)
        tick = time.time()
        opt = SolverFactory('gurobi')
        opt.options['NonConvex'] = 2
        opt.solve(model)
        tock = time.time()
        # round variables
        x = {}
        for j in range(num_var):
            x[j] = np.round(value(model.x[j]))
        # violation
        violations = 0
        for i in range(num_ineq):
            violations += max(sum([model.w[i,j]*x[j] for j in range(num_var)]) - model.c[i],0)
        obj = sum([-1*x[j] * model.p[j] for j in range(num_var)]) + sum([-1*model.q[j1,j2]*x[j1]*x[j2] for j1 in range(num_var-1) for j2 in range(j1+1,num_var)])
        objvals.append(obj)
        conviols.append(violations)
        elapseds.append(tock - tick)

        counter+=1
        print('Problem %i Solved (%f sec).'%(counter, tock - tick))


    print('-'*70)
    print('Summary results of run_EX_synergy for problems of the size of %i * %i' % (num_var, num_ineq))
    print('-'*70)


    df = pd.DataFrame({"Obj Val": objvals, "Constraints Viol": conviols, "Elapsed Time": elapseds})

    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Constraints Viol"] > 0)))

    print('-'*70)





















