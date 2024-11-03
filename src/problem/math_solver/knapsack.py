import numpy as np
from pyomo import environ as pe

from src.problem.math_solver import abcParamSolver


# ==============================================================================================
from dataclasses import dataclass
from typing import List, Optional, Union

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy.stats import uniform, randint
from scipy.stats.distributions import rv_frozen

from miplearn.io import read_pkl_gz
from miplearn.solvers.gurobi import GurobiModel


@dataclass
class MultiKnapsackData:
    """Data for the multi-dimensional knapsack problem

    Args
    ----
    prices
        Item prices.
    capacities
        Knapsack capacities.
    weights
        Matrix of item weights.
    """

    prices: np.ndarray
    capacities: np.ndarray
    weights: np.ndarray


# noinspection PyPep8Naming
class MultiKnapsackGenerator:
    """Random instance generator for the multi-dimensional knapsack problem.

    Instances have a random number of items (or variables) and a random number of
    knapsacks (or constraints), as specified by the provided probability
    distributions `n` and `m`, respectively. The weight of each item `i` on knapsack
    `j` is sampled independently from the provided distribution `w`. The capacity of
    knapsack `j` is set to ``alpha_j * sum(w[i,j] for i in range(n))``,
    where `alpha_j`, the tightness ratio, is sampled from the provided probability
    distribution `alpha`.

    To make the instances more challenging, the costs of the items are linearly
    correlated to their average weights. More specifically, the weight of each item
    `i` is set to ``sum(w[i,j]/m for j in range(m)) + K * u_i``, where `K`,
    the correlation coefficient, and `u_i`, the correlation multiplier, are sampled
    from the provided probability distributions. Note that `K` is only sample once
    for the entire instance.

    If `fix_w=True`, then `weights[i,j]` are kept the same in all generated
    instances. This also implies that n and m are kept fixed. Although the prices and
    capacities are derived from `weights[i,j]`, as long as `u` and `K` are not
    constants, the generated instances will still not be completely identical.

    If a probability distribution `w_jitter` is provided, then item weights will be
    set to ``w[i,j] * gamma[i,j]`` where `gamma[i,j]` is sampled from `w_jitter`.
    When combined with `fix_w=True`, this argument may be used to generate instances
    where the weight of each item is roughly the same, but not exactly identical,
    across all instances. The prices of the items and the capacities of the knapsacks
    will be calculated as above, but using these perturbed weights instead.

    By default, all generated prices, weights and capacities are rounded to the
    nearest integer number. If `round=False` is provided, this rounding will be
    disabled.

    Parameters
    ----------
    n: rv_discrete
        Probability distribution for the number of items (or variables).
    m: rv_discrete
        Probability distribution for the number of knapsacks (or constraints).
    w: rv_continuous
        Probability distribution for the item weights.
    K: rv_continuous
        Probability distribution for the profit correlation coefficient.
    u: rv_continuous
        Probability distribution for the profit multiplier.
    alpha: rv_continuous
        Probability distribution for the tightness ratio.
    fix_w: boolean
        If true, weights are kept the same (minus the noise from w_jitter) in all
        instances.
    w_jitter: rv_continuous
        Probability distribution for random noise added to the weights.
    round: boolean
        If true, all prices, weights and capacities are rounded to the nearest
        integer.
    """

    def __init__(
        self,
        n: rv_frozen = randint(low=100, high=101),
        m: rv_frozen = randint(low=30, high=31),
        w: rv_frozen = randint(low=0, high=1000),
        K: rv_frozen = randint(low=500, high=501),
        u: rv_frozen = uniform(loc=0.0, scale=1.0),
        alpha: rv_frozen = uniform(loc=0.25, scale=0.0),
        fix_w: bool = False,
        w_jitter: rv_frozen = uniform(loc=1.0, scale=0.0),
        p_jitter: rv_frozen = uniform(loc=1.0, scale=0.0),
        round: bool = True,
    ):
        assert isinstance(n, rv_frozen), "n should be a SciPy probability distribution"
        assert isinstance(m, rv_frozen), "m should be a SciPy probability distribution"
        assert isinstance(w, rv_frozen), "w should be a SciPy probability distribution"
        assert isinstance(K, rv_frozen), "K should be a SciPy probability distribution"
        assert isinstance(u, rv_frozen), "u should be a SciPy probability distribution"
        assert isinstance(
            alpha, rv_frozen
        ), "alpha should be a SciPy probability distribution"
        assert isinstance(fix_w, bool), "fix_w should be boolean"
        assert isinstance(
            w_jitter, rv_frozen
        ), "w_jitter should be a SciPy probability distribution"

        self.n = n
        self.m = m
        self.w = w
        self.u = u
        self.K = K
        self.alpha = alpha
        self.w_jitter = w_jitter
        self.p_jitter = p_jitter
        self.round = round
        self.fix_n: Optional[int] = None
        self.fix_m: Optional[int] = None
        self.fix_w: Optional[np.ndarray] = None
        self.fix_u: Optional[np.ndarray] = None
        self.fix_K: Optional[float] = None

        if fix_w:
            self.fix_n = self.n.rvs()
            self.fix_m = self.m.rvs()
            self.fix_w = np.array([self.w.rvs(self.fix_n) for _ in range(self.fix_m)])
            self.fix_u = self.u.rvs(self.fix_n)
            self.fix_K = self.K.rvs()

    def generate(self, n_samples: int) -> List[MultiKnapsackData]:
        def _sample() -> MultiKnapsackData:
            if self.fix_w is not None:
                assert self.fix_m is not None
                assert self.fix_n is not None
                assert self.fix_u is not None
                assert self.fix_K is not None
                n = self.fix_n
                m = self.fix_m
                w = self.fix_w
                u = self.fix_u
                K = self.fix_K
            else:
                n = self.n.rvs()
                m = self.m.rvs()
                w = np.array([self.w.rvs(n) for _ in range(m)])
                u = self.u.rvs(n)
                K = self.K.rvs()
            w = w * np.array([self.w_jitter.rvs(n) for _ in range(m)])
            alpha = self.alpha.rvs(m)
            p = np.array(
                [w[:, j].sum() / m + K * u[j] for j in range(n)]
            ) * self.p_jitter.rvs(n)
            b = np.array([w[i, :].sum() * alpha[i] for i in range(m)])
            if self.round:
                p = p.round()
                b = b.round()
                w = w.round()
            return MultiKnapsackData(p, b, w)

        return [_sample() for _ in range(n_samples)]
    
# ==============================================================================================


# add this to generate data
from scipy.stats import uniform, randint

class knapsack(abcParamSolver):
    def __init__(self, num_var, num_ineq, timelimit=None):
        super().__init__(timelimit=timelimit, solver="gurobi")

        # # generate a sample to fix prices and weigths
        # data = MultiKnapsackGenerator(
        #         n=randint(low=num_var, high=num_var+1),
        #         m=randint(low=num_ineq, high=num_ineq+1),
        #         w=uniform(loc=0, scale=1000),
        #         K=uniform(loc=100, scale=0),
        #         u=uniform(loc=1, scale=0),
        #         alpha=uniform(loc=0.25, scale=0),
        #         w_jitter=uniform(loc=0.95, scale=0.1),
        #         p_jitter=uniform(loc=0.75, scale=0.5),
        #         fix_w=True,
        #         ).generate(1)[0]

        rng = np.random.RandomState(17)
        raw_p = 0.1 * rng.random(num_var)   # prices
        raw_w = rng.uniform(0.5, 1.5, size=(num_ineq, num_var))  # weights

        # convert lists to dict
        p = {} # prices
        w = {} # weights

        for j in range(num_var):
            p[j] = raw_p[j]

        for i in range(num_ineq):
            for j in range(num_var):
                w[i,j] =raw_w[i][j]

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
        obj = sum(-1*m.x[j] * m.p[j]  for j in range(num_var))
        m.obj = pe.Objective(sense=pe.minimize, expr=obj)
        # constraints
        m.cons = pe.ConstraintList()
        for i in range(num_ineq):
            m.cons.add( sum([m.w[i,j]*m.x[j] for j in range(num_var)]) <= m.c[i] )

        m.pprint()

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

    # # generate sample of capacities (c)
    # data = MultiKnapsackGenerator(
    #             n=randint(low=num_var, high=num_var+1),
    #             m=randint(low=num_ineq, high=num_ineq+1),
    #             w=uniform(loc=0, scale=1000),
    #             K=uniform(loc=100, scale=0),
    #             u=uniform(loc=1, scale=0),
    #             alpha=uniform(loc=0.25, scale=0),
    #             w_jitter=uniform(loc=0.95, scale=0.1),
    #             p_jitter=uniform(loc=0.75, scale=0.5),
    #             fix_w=True,
    #             ).generate(num_data)
    
    # c = np.array([data[i].capacities for i in range(num_data)])
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
