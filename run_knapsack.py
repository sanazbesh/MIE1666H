import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import neuromancer as nm
from tqdm import tqdm

from src.problem import nmKnapsack, msKnapsack
from src.func.layer import netFC
from src.func import roundGumbelModel, roundThresholdModel

# random seed
np.random.seed(42)
torch.manual_seed(42)

def set_components(method, num_var, num_ineq, hlayers_sol, hlayers_rnd, hwidth):
    """
    Set components for NN model with rounding correction
    """
    # build neural architecture for the solution map specific to knapsack problem
    func = nm.modules.blocks.MLP(insize=num_ineq, outsize=num_var, bias=True,
                                 linear_map=nm.slim.maps["linear"],
                                 nonlin=nn.ReLU, hsizes=[hwidth]*hlayers_sol)
    smap = nm.system.Node(func, ["c"], ["x"], name="smap")

    # define rounding model
    layers_rnd = netFC(input_dim=num_ineq + num_var, hidden_dims=[hwidth]*hlayers_rnd, output_dim=num_var)

    if method == "classfication":
        rnd = roundThresholdModel(layers=layers_rnd, param_keys=["c"], var_keys=["x"], output_keys=["x_rnd"],
                                  int_ind={"x": range(num_var)}, continuous_update=True, name="round")
    else:
        rnd = roundGumbelModel(layers=layers_rnd, param_keys=["c"], var_keys=["x"], output_keys=["x_rnd"],
                               int_ind={"x": range(num_var)}, continuous_update=True, name="round")

    components = nn.ModuleList([smap, rnd])
    return components

def eval(data_test, model, components, num_var):
    """
    Evaluate model performance for knapsack problem
    """
    params, sols, objvals, conviols, elapseds = [], [], [], [], []
    for c in tqdm(data_test.datadict["c"][:100]):
        # data point as tensor
        datapoints = {"c": torch.unsqueeze(c, 0).to("cpu"), "name": "test"}
        components.eval()
        tick = time.time()
        with torch.no_grad():
            for comp in components:
                datapoints.update(comp(datapoints))
        tock = time.time()
        # assign parameters
        model.set_param_val({"c": c.cpu().numpy()})
        # assign variables
        x = datapoints["x_rnd"]
        for i in range(num_var):
            model.vars["x"][i].value = x[0, i].item()
        # get solutions
        xval, objval = model.get_val()
        params.append(list(c.cpu().numpy()))
        sols.append(list(list(xval.values())[0].values()))
        objvals.append(objval)
        conviols.append(sum(model.cal_violation()))
        elapseds.append(tock - tick)
    df = pd.DataFrame({"Param": params, "Sol": sols, "Obj Val": objvals, "Constraints Viol": conviols, "Elapsed Time": elapseds})
    time.sleep(1)
    print(df.describe())
    print("Number of infeasible solutions: {}".format(np.sum(df["Constraints Viol"] > 0)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--round", type=str, default="classfication", choices=["classfication", "threshold"],
                        help="method for rounding")
    parser.add_argument("--size", type=int, default=50, choices=[5, 10, 20, 50, 100, 200, 500],
                        help="number of decision variables and constraints")

    config = parser.parse_args()

    num_var = config.size     # number of variables (items)
    num_ineq = config.size    # number of constraints (knapsack capacities)
    num_data = 10000          # number of data
    test_size = 1000          # number of test size
    val_size = 1000           # number of validation size
    train_size = num_data - test_size - val_size

    # data sample from uniform distribution
    c_samples = torch.from_numpy(np.random.uniform(5, 10, size=(num_data, num_ineq))).float()
    data = {"c": c_samples}
    
    from src.utlis import data_split
    data_train, data_test, data_dev = data_split(data, test_size=test_size, val_size=val_size)

    from torch.utils.data import DataLoader
    batch_size = 64
    loader_train = DataLoader(data_train, batch_size, num_workers=0, collate_fn=data_train.collate_fn, shuffle=True)
    loader_test  = DataLoader(data_test, batch_size, num_workers=0, collate_fn=data_test.collate_fn, shuffle=False)
    loader_dev   = DataLoader(data_dev, batch_size, num_workers=0, collate_fn=data_dev.collate_fn, shuffle=False)

    # random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # hyperparameters
    hsize_dict = {5: 16, 10: 32, 20: 64, 50: 128, 100: 256, 200: 512, 500: 1024}
    penalty_weight = 100          
    hlayers_sol = 5               
    hlayers_rnd = 4               
    hwidth = hsize_dict[num_var]
    lr = 1e-3                     

    # get components for knapsack
    components = set_components(config.round, num_var, num_ineq, hlayers_sol, hlayers_rnd, hwidth)

    # loss function with constraint penalty for knapsack
    loss_fn = nmKnapsack(["c", "x_rnd"], num_var, num_ineq, penalty_weight=100)

    # training
    from src.problem.neuromancer.trainer import trainer
    epochs = 5
    warmup = 20
    patience = 20
    optimizer = torch.optim.AdamW(components.parameters(), lr=lr)
    my_trainer = trainer(components, loss_fn, optimizer, epochs, patience, warmup, device="cpu")
    my_trainer.train(loader_train, loader_dev)

    # evaluate
    model = msKnapsack(num_var, num_ineq)
    eval(data_test, model, components, num_var)
