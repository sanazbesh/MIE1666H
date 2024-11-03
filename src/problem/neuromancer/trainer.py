"""
Training pipeline
"""

import time

import copy
import torch

class trainer:
    def __init__(self, components, loss_fn, optimizer, epochs=100,
                 patience=5, warmup=0, clip=100, loss_key="loss", device="cpu"):
        """
        Initialize the Trainer class.
        """
        self.components = components
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.patience = patience
        self.warmup = warmup
        self.clip = clip
        self.loss_key = loss_key
        self.device = device
        self.early_stop_counter = 0
        self.best_loss = float("inf")
        self.best_model_state = None

    def train(self, loader_train, loader_dev):
        """
        Perform training with early stopping.
        """
        # initial validation loss calculation
        self.components.eval()
        with torch.no_grad():
            val_loss = self.best_loss = self.calculate_loss(loader_dev)
        # training loop
        tick = time.time()
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}, Validation Loss: {val_loss:.2f}")
            # training phase
            self.components.train()
            for data_dict in loader_train:
                # move to device
                for key in data_dict:
                    if torch.is_tensor(data_dict[key]):
                        data_dict[key] = data_dict[key].to(self.device)
                # forwad pass
                for comp in self.components:
                    data_dict.update(comp(data_dict))
                if 'x_rnd' in data_dict:
                    data_dict['x_rnd'] = torch.clamp(data_dict['x_rnd'], min=0)
                data_dict = self.loss_fn(data_dict)
                # backward pass
                data_dict[self.loss_key].backward()
                torch.nn.utils.clip_grad_norm_(self.components.parameters(), self.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
            # validation phase
            self.components.eval()
            with torch.no_grad():
                val_loss = self.calculate_loss(loader_dev)
            # early stopping check
            if epoch >= self.warmup:
                early_stop = self.update_early_stopping(val_loss)
                if self.early_stop_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        tock = time.time()
        elapsed = tock - tick
        # end of training
        if self.best_model_state:
            self.components.load_state_dict(self.best_model_state)
            print("Best model loaded.")
        print("Training complete.")
        print(f"The training time is {elapsed:.2f} sec.")

    def calculate_loss(self, loader):
        """
        Calculate loss for a given dataset loader.
        """
        total_loss = 0.0
        for data_dict in loader:
            # move to device
            for key in data_dict:
                if torch.is_tensor(data_dict[key]):
                    data_dict[key] = data_dict[key].to(self.device)
            # forward pass
            for comp in self.components:
                data_dict.update(comp(data_dict))
            total_loss += self.loss_fn(data_dict)[self.loss_key].item()
        return total_loss / len(loader)

    def update_early_stopping(self, val_loss):
        """
        Update the early stopping counter and model state.
        """
        # update with better loss
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(self.components.state_dict())
            self.early_stop_counter = 0  # reset early stopping counter
        else:
            self.early_stop_counter += 1
