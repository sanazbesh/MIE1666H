import numpy as np
import torch
from scipy.stats import uniform, randint
from src.problem.math_solver import abcParamSolver
from src.problem.math_solver.KnapsackGenerator import MultiKnapsackGenerator

from src.problem import msKnapsack_synergy

# random seed
np.random.seed(42)
torch.manual_seed(42)

if __name__ == "__main__":

    from src.utlis import ms_test_solve


    num_var = 5
    num_ineq = 10
    num_data = 5000

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
    raw_w = an_instance.weights
    max_weights = [max(row) for row in raw_w]
    scaled_c = [25 * max_weight for max_weight in max_weights]
    
    #c_samples = np.random.uniform(100, 300, size=(num_data, num_ineq))
    # set params
    params = {"c":scaled_c}
    # init model
    model = msKnapsack_synergy(num_var, num_ineq)

    # solve the MIQP
    print("======================================================")
    print("Solve MINLP problem:")
    model.set_param_val(params)
    ms_test_solve(model)

    

