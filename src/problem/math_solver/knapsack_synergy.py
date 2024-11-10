import numpy as np
from pyomo import environ as pe

from scipy.stats import uniform, randint
from src.problem.math_solver import abcParamSolver
from src.problem.math_solver.KnapsackGenerator import MultiKnapsackGenerator

class knapsack_synergy(abcParamSolver):
    def __init__(self, num_var, num_ineq, timelimit=None):
        super().__init__(timelimit=timelimit, solver="gurobi")

        an_instance = MultiKnapsackGenerator(
                n=randint(low=num_var, high=num_var+1),
                m=randint(low=num_ineq, high=num_ineq+1),
                w=uniform(loc=0, scale=60),
                K=uniform(loc=100, scale=0),
                u=uniform(loc=1, scale=0),
                alpha=uniform(loc=0.25, scale=0),
                w_jitter=uniform(loc=0.95, scale=0.1),
                p_jitter=uniform(loc=0.75, scale=0.5),
                rng_state=17
            ).generate(1)[0]

        raw_p = an_instance.prices
        raw_w = an_instance.weights

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
        # mutable parameters (parametric part of the problem)
        m.c = pe.Param(pe.RangeSet(0,num_ineq-1), default=25*max([max(raw_w[i]) for i in range(len(raw_w))]), mutable=True)
        # decision variables
        m.x = pe.Var(pe.RangeSet(0,num_var-1), domain=pe.NonNegativeIntegers)
        # objective function
        obj = sum([-1*m.x[j] * m.p[j]  for j in range(num_var)]) + sum([-1*q[j1,j2]*m.x[j1]*m.x[j2] for j1 in range(num_var-1) for j2 in range(j1+1,num_var)])
        m.obj = pe.Objective(sense=pe.minimize, expr=obj)
        # constraints
        m.cons = pe.ConstraintList()
        for i in range(num_ineq):
            m.cons.add( sum([m.w[i,j]*m.x[j] for j in range(num_var)]) <= m.c[i] )

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

    c_samples = np.random.uniform(5, 10, size=(num_data, num_ineq))
    # set params
    params = {"c":c_samples[0]}
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
