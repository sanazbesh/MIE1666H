"""
Parametric Mixed Integer Quadratic Programming

https://www.sciencedirect.com/science/article/pii/S0098135401007979
https://www.sciencedirect.com/science/article/pii/S1570794601801577
"""

import numpy as np
import torch
from torch import nn
import neuromancer as nm

class penaltyLoss(nn.Module):
    """
    Penalty loss function for quadratic problem
    """
    def __init__(self, input_keys, Q, p, A, penalty_weight=50, output_key="loss"):
        super().__init__()
        self.b_key, self.x_key = input_keys
        self.output_key = output_key
        self.penalty_weight = penalty_weight
        # fixed coefficients
        self.Q = Q
        self.p = p
        self.A = A

    def forward(self, input_dict):
        """
        forward pass
        """
        # objective function
        obj = self.cal_obj(input_dict)
        # constraints violation
        viol = self.cal_constr_viol(input_dict)
        # penalized loss
        loss = obj + self.penalty_weight * viol
        input_dict[self.output_key] = torch.mean(loss)
        return input_dict

    def cal_obj(self, input_dict):
        """
        calculate objective function
        """
        # get values
        x = input_dict[self.x_key]
        # 1/2 x^T Q x
        Q_term = torch.einsum("bm,nm,bm->b", x, self.Q, x) / 2
        # p^T y
        p_term = torch.einsum("m,bm->b", self.p, torch.sin(x))
        return Q_term + p_term

    def cal_constr_viol(self, input_dict):
        """
        calculate constraints violation
        """
        # get values
        x, b = input_dict[self.x_key], input_dict[self.b_key]
        # constraints
        lhs = torch.einsum("mn,bn->bm", self.A, x) # Ax
        rhs = b # b
        violation = torch.relu(lhs - rhs).sum(dim=1) # Ax<=b
        return violation


if __name__ == "__main__":

    # random seed
    np.random.seed(17)
    torch.manual_seed(17)

    # init
    num_var = 10      # numner of variables
    num_ineq = 10     # numner of constraints
    num_data = 5000   # number of data
    test_size = 1000  # number of test size
    val_size = 1000   # number of validation size

    # generate parameters
    Q = torch.from_numpy(np.diag(np.random.random(size=num_var))).float()
    p = torch.from_numpy(np.random.random(num_var)).float()
    A = torch.from_numpy(np.random.normal(scale=1, size=(num_ineq, num_var))).float()

    # data sample from uniform distribution
    b_samples = torch.from_numpy(np.random.uniform(-1, 1, size=(num_data, num_ineq))).float()
    data = {"b":b_samples}
    # data split
    from src.utlis import data_split
    data_train, data_test, data_dev = data_split(data, test_size=test_size, val_size=val_size)
    # torch dataloaders
    from torch.utils.data import DataLoader
    loader_train = DataLoader(data_train, batch_size=32, num_workers=0,
                              collate_fn=data_train.collate_fn, shuffle=True)
    loader_test  = DataLoader(data_test, batch_size=32, num_workers=0,
                              collate_fn=data_test.collate_fn, shuffle=True)
    loader_dev   = DataLoader(data_dev, batch_size=32, num_workers=0,
                              collate_fn=data_dev.collate_fn, shuffle=True)

    # define neural architecture for the solution map smap(p) -> x
    import neuromancer as nm
    func = nm.modules.blocks.MLP(insize=num_ineq, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[10]*4)
    components = nn.ModuleList([nm.system.Node(func, ["b"], ["x"], name="smap")])

    # build neuromancer problem
    loss_fn = penaltyLoss(["b", "x"], Q, p, A)

    # training
    lr = 0.001    # step size for gradient descent
    epochs = 200  # number of training epochs
    warmup = 20   # number of epochs to wait before enacting early stopping policy
    patience = 20 # number of epochs with no improvement in eval metric to allow before early stopping
    # set adamW as optimizer
    optimizer = torch.optim.AdamW(components.parameters(), lr=lr)
    # training
    from src.problem.neuromancer.trainer import trainer
    my_trainer = trainer(components, loss_fn, optimizer, epochs, patience, warmup)
    my_trainer.train(loader_train, loader_dev)
    print()

    # init mathmatic model
    from src.problem.math_solver.nonconvex import nonconvex
    model = nonconvex(Q.cpu().numpy(), p.cpu().numpy(), A.cpu().numpy())

    # test neuroMANCER
    from src.utlis import nm_test_solve
    print("neuroMANCER:")
    datapoint = {"b": b_samples[:1],
                 "name":"test"}
    model.set_param_val({"b":b_samples[0].cpu().numpy()})
    nm_test_solve("x", components, datapoint, model)
