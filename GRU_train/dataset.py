import warnings
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
import os

class WindTurbineData:
    """
    Desc: Data preprocessing,
          Here, e.g.    15 days for training,
                        3 days for validation,
                        and 6 days for testing
    """
    def __init__(self, data_path,
                 filename='sdwpf_baidukddcup2022_full.CSV',
                 flag='train',
                 size=None,
                 turbine_id=0,
                 task='MS',
                 target='Target',
                 scale=True,
                 start_col=3,       # the start column index of the data one aims to utilize
                 day_len=24 * 6,
                 train_days=15,     # 15 days
                 val_days=3,        # 3 days
                 test_days=6,       # 6 days
                 total_days=30      # 30 days
                 ):
        super().__init__()
        self.unit_size = day_len
        if size is None:
            self.input_len = self.unit_size
            self.output_len = self.unit_size
        else:
            self.input_len = size[0]
            self.output_len = size[1]
        # initialization
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.task = task
        self.target = target
        self.scale = scale
        self.start_col = start_col
        self.data_path = data_path
        self.filename = filename
        self.tid = turbine_id
        self.transform = transforms.ToTensor()
        # If needed, we employ the predefined total_size (e.g. one month)
        from_days = 0
        self.from_size = from_days * self.unit_size
        self.total_size = self.unit_size * total_days
        #
        self.train_size = train_days * self.unit_size
        self.val_size = val_days * self.unit_size
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.data_path, self.filename))
        border1s = [self.tid * self.total_size + self.from_size,
                    self.tid * self.total_size + self.from_size + self.train_size - self.input_len
                    ]
        border2s = [self.tid * self.total_size + self.from_size + self.train_size,
                    self.tid * self.total_size + self.from_size + self.train_size + self.val_size
                    ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.task == 'M':
            cols_data = df_raw.columns[self.start_col:]
            df_raw = df_raw[cols_data]
        elif self.task == 'MS':
            cols_data = df_raw.columns[self.start_col:]
            df_raw = df_raw[cols_data]
        elif self.task == 'S':
            df_raw = df_raw[[self.tid, self.target]]

        df_data = df_raw
        # Turn off the SettingWithCopyWarning
        pd.set_option('mode.chained_assignment', None)
        df_data.replace(to_replace=np.nan, value=0, inplace=True)

        data = df_data.values
        #####################
        # transform
        #####################
        mean = np.loadtxt('mean.txt',  dtype=np.float32)
        std = np.loadtxt('std.txt', dtype=np.float32)
        data = (data - mean[self.tid]) / std[self.tid]
        flag = np.loadtxt('flag.txt', dtype=np.int16)
        valid = flag[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        indices = np.where(valid == -1)
        self.data_y[indices, -1] = -1e5
        ######################
        # raw_data same as data_y
        ######################
        self.raw_data = df_raw[border1 + self.input_len:border2]

    def get_raw_data(self):
        return self.raw_data

    def __getitem__(self, index):
        #
        # Only for customized use.
        # When sliding window not used, e.g. prediction without overlapped input/output sequences
        if self.set_type >= 3:
            index = index * self.output_len
        #
        # Standard use goes here.
        # Sliding window with the size of input_len + output_len
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # transform to tensors
        seq_x = torch.from_numpy(seq_x).float()
        seq_y = torch.from_numpy(seq_y).float()
        return seq_x, seq_y

    def __len__(self):
        # In our case, the sliding window is adopted, the number of samples is calculated as follows
        if self.set_type < 3:
            return len(self.data_x) - self.input_len - self.output_len + 1
        # Otherwise, if sliding window is not adopted
        return int((len(self.data_x) - self.input_len) / self.output_len)
