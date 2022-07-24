from typing import Callable
import time
import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
import numpy as np
from model import BaselineGruModel
from dataset import WindTurbineData

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.mae = nn.L1Loss(reduction='mean')
    def forward(self, sample, truth):
        out_rmse = torch.sqrt(self.mse(sample, truth))
        out_mae = self.mae(sample, truth)
        return 0.5 * out_rmse + 0.5 * out_mae

def adjust_learning_rate(optimizer, epoch, args):
    # type: (torch.optimizer.Adam, int, dict) -> None
    """
    Desc:
        Adjust learning rate
    Args:
        optimizer:
        epoch:
        args:
    Returns:
        None
    """
    lr_adjust = {}
    if args["lr_adjust"] == 'type1':
        lr_adjust = {epoch: args["lr"] * (0.50 ** (epoch - 1))}
    elif args["lr_adjust"] == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups: 
            param_group['lr'] = lr


class EarlyStopping(object):
    """
    Desc:
        EarlyStopping
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = False

    def save_checkpoint(self, val_loss, model, path, tid):
        """
        Desc:
            Save current checkpoint
        Args:
            val_loss: the validation loss
            model: the model
            path: the path to be saved
            tid: turbine ID
        Returns:
            None
        """
        self.best_model = True
        self.val_loss_min = val_loss
        torch.save(model.state_dict(), path + '/' + 'model_' + str(tid))

    def __call__(self, val_loss, model, path, tid):
        """
        Desc:
            __call__
        Args:
            val_loss: the validation loss
            model: the model
            path: the path to be saved
            tid: turbine ID
        Returns:
            None
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, tid)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.best_model = False
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.update_hidden = True
            self.save_checkpoint(val_loss, model, path, tid)
            self.counter = 0


class Experiment(object):
    """
    Desc:
        The experiment to train, validate and test a model
    """
    def __init__(self, args):
        """
        Desc:
            __init__
        Args:
            args: the arguments to initialize the experimental environment
        """
        self.args = args

    def get_args(self):
        """
        Desc:
            Get the arguments
        Returns:
            A dict
        """
        return self.args

    def get_data(self, flag):
        """
        Desc:
            get_data
        Args:
            flag: train or val
        Returns:
            A dataset and a dataloader
        """
        data_set = WindTurbineData(
            data_path=self.args["data_path"],
            filename=self.args["filename"],
            flag=flag,
            size=[self.args["input_len"], self.args["output_len"]],
            task=self.args["task"],
            target=self.args["target"],
            start_col=self.args["start_col"],
            turbine_id=self.args["turbine_id"],
            day_len=self.args["day_len"],
            train_days=self.args["train_size"],
            val_days=self.args["val_size"],
            total_days=self.args["total_size"]
        )
        data_loader = DataLoader(
            data_set,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args["num_workers"]
        )
        return data_set, data_loader

    def get_optimizer(self, model):
        """
        Desc:
            Get the optimizer
        Returns:
            An optimizer
        """
        model_optim = torch.optim.Adam(model.parameters(), self.args["lr"], weight_decay=5e-4)
        return model_optim

    @staticmethod
    def get_criterion():
        """
        Desc:
            Use the mse loss as the criterion
        Returns:
            MSE loss
        """
        # criterion = nn.MSELoss(reduction='mean')
        criterion = Loss()
        return criterion

    def process_one_batch(self, model, batch_x, batch_y):
        """
        Desc:
            Process a batch
        Args:
            model:
            batch_x:
            batch_y:
        Returns:
            prediction and ground truth
        """

        sample = model(batch_x)
        f_dim = -1 if self.args["task"] == 'MS' else 0
        
        batch_y = batch_y[:,-self.args["output_len"]:, f_dim:].type(torch.float32)
        indice = (batch_y != -1e5)
        batch_y = batch_y[indice]
        ##########################
        # delete abnormal
        ##########################
        sample = sample[..., :, f_dim:].type(torch.float32)
        sample = sample[indice]
        return sample, batch_y