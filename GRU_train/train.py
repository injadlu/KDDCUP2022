import os
import time
import numpy as np
from typing import Callable
import torch
import random
from torch.utils.data import DataLoader
from model import BaselineGruModel
from common import EarlyStopping
from common import adjust_learning_rate
from common import Experiment
from prepare import prep_env
import torch.nn as nn

def val(experiment, model, data_loader, criterion):
    """
    Desc:
        Validation function
    Args:
        experiment:
        model:
        data_loader:
        criterion:
    Returns:
        The validation loss
    """
    validation_loss = []
    args = experiment.get_args()
    for i, (batch_x, batch_y) in enumerate(data_loader):
        if args["use_gpu"]:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        sample, true = experiment.process_one_batch(model, batch_x, batch_y)
        loss = criterion(sample, true)
        validation_loss.append(loss.item())
    validation_loss = np.average(validation_loss)
    return validation_loss


def train_and_val(experiment, model, model_folder, is_debug=False):
    """
    Desc:
        Training and validation
    Args:
        experiment:
        model:
        model_folder: folder name of the model
        is_debug:
    Returns:
        None
    """
    args = experiment.get_args()
    train_data, train_loader = experiment.get_data(flag='train')
    val_data, val_loader = experiment.get_data(flag='val')

    path_to_model = os.path.join(args["checkpoints"], model_folder)
    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)

    early_stopping = EarlyStopping(patience=args["patience"], verbose=True)
    optimizer = experiment.get_optimizer(model)
    criterion = Experiment.get_criterion()

    epoch_start_time = time.time()
    if args["use_gpu"]:
        model = model.cuda()
    for epoch in range(args["train_epochs"]):
        iter_count = 0
        train_loss = []
        model.train()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            if args["use_gpu"]:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            iter_count += 1
            optimizer.zero_grad()
            sample, truth = experiment.process_one_batch(model, batch_x, batch_y)
            loss = criterion(sample, truth)
            train_loss.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0, norm_type=2)
            optimizer.step()
            
        model.eval()
        val_loss = val(experiment, model, val_loader, criterion)
        if is_debug:
            train_loss = np.average(train_loss)
            epoch_end_time = time.time()
            print("Epoch: {}, \nTrain Loss: {}, \nValidation Loss: {}".format(epoch, train_loss, val_loss))
            print("Elapsed time for epoch-{}: {}".format(epoch, epoch_end_time - epoch_start_time))
            epoch_start_time = epoch_end_time

        # Early Stopping if needed
        early_stopping(val_loss, model, path_to_model, args["turbine_id"])
        if early_stopping.early_stop:
            print("Early stopped! ")
            break
        adjust_learning_rate(optimizer, epoch + 1, args)


if __name__ == "__main__":
    fix_seed = 3407
    random.seed(fix_seed)
    np.random.seed(fix_seed)

    settings = prep_env()
    cur_setup = 'GRU_ALL'
    start_train_time = time.time()
    end_train_time = start_train_time
    start_time = start_train_time
    for tid in range(settings["turbine_id"], settings["capacity"]):
        settings["turbine_id"] = tid
        exp = Experiment(settings)
        print('\n>>>>>>> Training Turbine {:3d} >>>>>>>>>>>>>>>>>>>>>>>>>>\n'.format(tid))
        baseline_model = BaselineGruModel(settings)
        train_and_val(exp, model=baseline_model, model_folder=cur_setup, is_debug=settings["is_debug"])
        torch.cuda.empty_cache()
        if settings["is_debug"]:
            end_time = time.time()
            print("\nTraining the {}-th turbine in {} secs".format(tid, end_time - start_time))
            start_time = end_time
            end_train_time = end_time
    if settings["is_debug"]:
        print("\nTotal time in training {} turbines is "
              "{} secs".format(settings["capacity"], end_train_time - start_train_time))
