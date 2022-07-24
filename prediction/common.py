from typing import Callable
import time
import numpy as np
import pandas as pd
import torch
class TestData(object):
    """
        Desc: Test Data
    """
    def __init__(self,
                 path_to_data,
                 task='MS',
                 target='Patv',
                 start_col=3,       # the start column index of the data one aims to utilize
                 farm_capacity=134
                 ):
        self.task = task
        self.target = target
        self.start_col = start_col
        self.data_path = path_to_data
        self.farm_capacity = farm_capacity
        self.df_data = pd.read_csv(self.data_path)
        self.total_size = int(self.df_data.shape[0] / self.farm_capacity)
        # Handling the missing values
        self.df_data.replace(to_replace=np.nan, value=0, inplace=True)

    def get_turbine(self, tid):
        begin_pos = tid * self.total_size
        border1 = begin_pos
        border2 = begin_pos + self.total_size
        if self.task == 'MS':
            cols = self.df_data.columns[self.start_col:]
            # cols = ["Patv","Etmp","Itmp","Wspd"]
            data = self.df_data[cols]
        else:
            raise Exception("Unsupported task type ({})! ".format(self.task))
        seq = data.values[border1:border2]
        return seq

    def get_all_turbines(self):
        seqs = []
        for i in range(self.farm_capacity):
            seq = self.get_turbine(i)
            seqs.append(seq)
            # dfs.append(df)
        return seqs
    def get_test_data(self, min, max):
        input_len = 144
        test_data = []
        cols = self.df_data.columns[self.start_col:]
        data = self.df_data[cols]
        data = data.values
        for tid in range(0, 134):
            border_1 = (tid+1) * self.total_size - input_len
            border_2 = (tid+1) * self.total_size
            data[border_1 : border_2] = (data[border_1 : border_2] - min[tid]) / (max[tid] - min[tid])
            test_data.append(data[border_1 : border_2])
        test_data = np.array(test_data)
        return test_data
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


    @staticmethod
    def get_test_x(args):
        """
        Desc:
            Obtain the input sequence for testing
        Args:
            args:
        Returns:
            Normalized input sequences and training data
        """
        test_x = TestData(path_to_data=args["path_to_test_x"], farm_capacity=args["capacity"])
        return test_x
    def inference_one_sample_from_lstm(self, model, sample_x):
        """
        Desc:
            Inference one sample
        Args:
            model:
            sample_x:
        Returns:
            Predicted sequence with sample_x as input
        """
        # predifined when load data
        # x = sample_x.astype('float32')
        prediction = model(sample_x)
        f_dim = -1 if self.args["task"] == 'MS' else 0
        # (1, 288, 1)
        return prediction[..., :, f_dim:].type(torch.float32)