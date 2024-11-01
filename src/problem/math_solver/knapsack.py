import numpy as np
from pyomo import environ as pe

from src.problem.math_solver import abcParamSolver
from src.problem.math_solver import MultiKnapsackGenerator

# add this to generate data
from scipy.stats import uniform, randint

class knapsack(abcParamSolver):
    def __init__(self, num_var, num_ineq, timelimit=None):
        super().__init__(timelimit=timelimit, solver="gurobi")

        # generate a sample to fix prices and weigths
        """
        I am not sure if this is a good way to generate instances,
        but this is similar to quadratic function
        where they generate fixed params A,Q,p (p,w in our problem)
        and set the right hand side b (c in our problem) to zero: m.c = pe.Param(pe.RangeSet(num_ineq), default=0, mutable=True)
        to change it outside the function
        
        notice that m.c is mutuable which means it can be changed
        we can change the default to a larger value, but I don't know if it matters
        """
        data = MultiKnapsackGenerator(
                n=randint(low=num_var, high=num_var+1),
                m=randint(low=num_ineq, high=num_ineq+1),
                w=uniform(loc=0, scale=1000),
                K=uniform(loc=100, scale=0),
                u=uniform(loc=1, scale=0),
                alpha=uniform(loc=0.25, scale=0),
                w_jitter=uniform(loc=0.95, scale=0.1),
                p_jitter=uniform(loc=0.75, scale=0.5),
                fix_w=True,
                ).generate(1)[0]
        
        # convert lists to dict
        p = {} # prices
        w = {} # weights
        c = {} # capacities

        for j in range(1, num_var+1):
            p[j] = data.prices[j-1]

        for i in range(1, num_ineq+1):
            c[i] = data.capacities[i-1]
            for j in range(1, num_var+1):
                w[i,j] = data.weights[i-1][j-1]
        
        # fixed params are p and w
        
        # size
        num_ineq = len(w)
        num_var = len(p)

        # Mathematical model
        m = pe.ConcreteModel()
        # mutable parameters (parametric part of the problem)
        m.c = pe.Param(pe.RangeSet(num_ineq), default=0, mutable=True)
        # decision variables
        m.x = pe.Var(pe.RangeSet(num_var), domain=pe.Binary)
        # objective function
        obj = sum(-1*m.x[j] * p[j]  for j in range(1,num_var+1))
        m.obj = pe.Objective(sense=pe.minimize, expr=obj)
        # constraints
        m.cons = pe.ConstraintList()
        for i in range(1,num_ineq+1):
            m.cons.add( sum([w[i,j]*m.x[j] for j in range(1,num_var+1)]) <= m.c[i] )

        # set attributes
        self.model = m
        self.params ={"c":m.c}
        self.vars = {"x":m.x}
        self.cons = m.cons


if __name__ == "__main__":

    from src.utlis import ms_test_solve

    num_var = 10
    num_ineq = 10
    num_data = 5000

    # generate sample of capacities (c)
    data = MultiKnapsackGenerator(
                n=randint(low=num_var, high=num_var+1),
                m=randint(low=num_ineq, high=num_ineq+1),
                w=uniform(loc=0, scale=1000),
                K=uniform(loc=100, scale=0),
                u=uniform(loc=1, scale=0),
                alpha=uniform(loc=0.25, scale=0),
                w_jitter=uniform(loc=0.95, scale=0.1),
                p_jitter=uniform(loc=0.75, scale=0.5),
                fix_w=True,
                ).generate(num_data)
    
    c = np.array([data[i].capacities for i in range(num_data)])
    # set params
    params = {"c":c[0]}
    # init model
    model = knapsack(num_var, num_ineq)

    # solve the MIQP
    print("======================================================")
    print("Solve MINLP problem:")
    model.set_param_val(params)
    ms_test_solve(model)

    # solve the penalty
    print()
    print("======================================================")
    print("Solve penalty problem:")
    model_pen = model.penalty(100)
    model_pen.set_param_val(params)
    # scip
    ms_test_solve(model_pen)

    # solve the relaxation
    print()
    print("======================================================")
    print("Solve relaxed problem:")
    model_rel = model.relax()
    model_rel.set_param_val(params)
    # scip
    ms_test_solve(model_rel)
