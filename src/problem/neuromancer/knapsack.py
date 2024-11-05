import numpy as np
import torch
from torch import nn
import neuromancer as nm


# Custom activation function to ensure non-negative integers
class NonNegativeIntegerActivation(nn.Module):
    def forward(self, x):
        return torch.floor(torch.clamp(x, min=0))  # Ensure non-negativity and integer

class penaltyLoss(nn.Module):
    """
    Penalty loss function for knapsack problem
    """
    def __init__(self, input_keys, num_var, num_ineq, penalty_weight=50, output_key="loss"):
        super().__init__()
        self.c_key, self.x_key = input_keys
        self.output_key = output_key
        self.penalty_weight = penalty_weight
        self.device = None
        # Fixed parameters
        from src.problem.math_solver.KnapsackGenerator import MultiKnapsackGenerator
        from scipy.stats import uniform, randint
        rng = np.random.RandomState(17)
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
        
        p = an_instance.prices   # prices
        w = an_instance.weights  # weights
        self.p = torch.from_numpy(p).float()
        self.w = torch.from_numpy(w).float()

    def forward(self, input_dict):
        """
        forward pass
        """

        # Objective function
        obj = self.cal_obj(input_dict)
        # Constraints violation
        viol = self.cal_constr_viol(input_dict)
        # Penalized loss
        loss = obj + self.penalty_weight * viol
        input_dict[self.output_key] = torch.mean(loss)
        return input_dict

    def cal_obj(self, input_dict):
        """
        Calculate objective function: negative total profit to maximize profit
        """
        x = input_dict[self.x_key]
        # Update device
        if self.device is None:
            self.device = x.device
            self.p = self.p.to(self.device)
            self.w = self.w.to(self.device)
        return -torch.einsum("m,bm->b", self.p, x)  # Maximize profit by minimizing negative profit

    def cal_constr_viol(self, input_dict):
        """
        Calculate constraints violation based on capacities and weights
        """
        x, c = input_dict[self.x_key], input_dict[self.c_key]
        lhs = torch.einsum("ij,bj->bi", self.w, x)  # w * x
        violation = torch.relu(lhs - c).sum(dim=1)   # Enforce w*x <= c
        return violation


if __name__ == "__main__":
    # Random seed
    np.random.seed(17)
    torch.manual_seed(17)

    # Init
    num_var = 10      # Number of variables (items)
    num_ineq = 10     # Number of constraints (knapsack capacities)
    num_data = 5000   # Number of data
    test_size = 1000  # Number of test size
    val_size = 1000   # Number of validation size

    # Generate data samples for capacities (c)
    c_samples = torch.from_numpy(np.random.uniform(30, 50, size=(num_data, num_ineq))).float()
    data = {"c": c_samples}
    # Data split
    from src.utlis import data_split
    data_train, data_test, data_dev = data_split(data, test_size=test_size, val_size=val_size)
    # Torch dataloaders
    from torch.utils.data import DataLoader
    loader_train = DataLoader(data_train, batch_size=32, num_workers=0,
                              collate_fn=data_train.collate_fn, shuffle=True)
    loader_test  = DataLoader(data_test, batch_size=32, num_workers=0,
                              collate_fn=data_test.collate_fn, shuffle=True)
    loader_dev   = DataLoader(data_dev, batch_size=32, num_workers=0,
                              collate_fn=data_dev.collate_fn, shuffle=True)

    # Define neural architecture for the solution map smap(c) -> x
    func = nm.modules.blocks.MLP(insize=num_ineq, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[10]*4)
    
    components = nn.ModuleList([nm.system.Node(func, ["c"], ["x"], name="smap")])

    # Build neuromancer problem
    loss_fn = penaltyLoss(["c", "x"], num_var, num_ineq)

    # Training
    lr = 0.001    # Step size for gradient descent
    epochs = 200  # Number of training epochs
    warmup = 20   # Number of epochs to wait before enacting early stopping policy
    patience = 20 # Number of epochs with no improvement in eval metric to allow before early stopping
    optimizer = torch.optim.AdamW(components.parameters(), lr=lr)
    from src.problem.neuromancer.trainer import trainer
    my_trainer = trainer(components, loss_fn, optimizer, epochs, patience, warmup)
    my_trainer.train(loader_train, loader_dev)
    print()

    # Init mathematical model
    from src.problem.math_solver.knapsack import knapsack
    model = knapsack(num_var, num_ineq)

    # Test neuroMANCER
    from src.utlis import nm_test_solve
    print("neuroMANCER:")
    datapoint = {"c": c_samples[:1],
                 "name":"test"}
    model.set_param_val({"c": c_samples[0].cpu().numpy()})
    nm_test_solve("x", components, datapoint, model)
