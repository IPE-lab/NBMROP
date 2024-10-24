from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset
import torch 
from typing import Any, Optional, Tuple
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

well_path_dict = {
    'well_0': 'USROP_A 0 N-NA_F-9_Ad.csv',
    'well_1': 'USROP_A 1 N-S_F-7d.csv',
    'well_2':  'USROP_A 2 N-SH_F-14d.csv',
    'well_3': 'USROP_A 3 N-SH-F-15d.csv',
    'well_4': 'USROP_A 4 N-SH_F-15Sd.csv',
    'well_5': 'USROP_A 5 N-SH-F-5d.csv',
    'well_6' : 'USROP_A 6 N-SH_F-9d.csv'
    }

# Measured Depth m,Weight on Bit kkgf,Average Standpipe Pressure kPa,Average Surface Torque kN.m,Rate of Penetration m/h,Average Rotary Speed rpm,Mud Flow In L/min,Mud Density In g/cm3,Diameter mm,Average Hookload kkgf,Hole Depth (TVD) m,USROP Gamma gAPI
total_columns = ['Measured Depth m', # 0
                'Weight on Bit kkgf', # 1
                'Average Standpipe Pressure kPa', # 2
                'Average Surface Torque kN.m', # 3
                'Rate of Penetration m/h', # 4
                'Average Rotary Speed rpm', # 5
                'Mud Flow In L/min', # 6
                'Mud Density In g/cm3', # 7
                'Diameter mm', # 8
                'Average Hookload kkgf', # 9
                'Hole Depth (TVD) m', # 10
                'USROP Gamma gAPI'] # 11

class ROPdataset(Dataset):
    def __init__(
        self,
        test_well: str,
        train_val_test: str,
        train_fold: int,
        sample_interval: int = 1,
        selected_columns = [0, 1, 2, 3, 4,5, 9, 11]
        ):
        # load_train_data
        data_folder = './dataset/processed_dataset/USROP/'
        train_data_paths = [os.path.join(data_folder, well_path_dict[well]) for well in well_path_dict.keys() if well != test_well]
        pds_list = [pd.read_csv(path) for path in train_data_paths]

        # pds_list = [pds.rolling(window=5).mean().dropna() for pds in pds_list]
        total_df = pd.concat(pds_list, ignore_index=True)
        del total_df['Unnamed: 0']

        if selected_columns is not None:
            selected_columns = [total_columns[i] for i in selected_columns]

        total_df = total_df[total_columns]
        total_df = total_df.iloc[::sample_interval, :]
        

        self.scaler = StandardScaler()
        self.scaler.fit(total_df)
        scalered_df = self.scaler.transform(total_df)
        scalered_df = pd.DataFrame(scalered_df, columns = total_df.columns)

        # split data
        kf = KFold(n_splits=5, shuffle=False, random_state=None)
        splited_kf = list(kf.split(scalered_df))
        train_idx, val_idx = splited_kf[train_fold]

        if train_val_test == 'train':
            self.data = scalered_df.iloc[train_idx, :]
        elif train_val_test == 'validation':
            self.data = scalered_df.iloc[val_idx, :]
        elif train_val_test == 'test':
            self.data = pd.read_csv(os.path.join(data_folder, well_path_dict[test_well]))
            self.data = self.data.iloc[::sample_interval, :]
            del self.data['Unnamed: 0']
            self.data = self.scaler.transform(self.data)
            self.data = pd.DataFrame(self.data, columns = total_df.columns)
        else:
            raise ValueError('train_val_test should be one of [train, validation, test]')
        
        self.target = self.data['Rate of Penetration m/h']
        del self.data['Rate of Penetration m/h']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.Tensor(self.data.iloc[idx, :].values), torch.Tensor([self.target.iloc[idx]])

        
        
class ROPContinousLearningDataset(Dataset):
    def __init__(
        self,
        target_well: str,
        train_val_test: str,
        rigind: int,
        ):
        # load_train_data
        train_data_path = './dataset/processed_dataset/USROP/' + well_path_dict[target_well]
        self.df = pd.read_csv(train_data_path)
        del self.df['Unnamed: 0']
        del self.df['Hole Depth (TVD) m']

        increment = 577
        depth = increment
        self.max_rigind = len(self.df) // increment - 2
        
        if rigind > self.max_rigind:
            raise ValueError(f'rigind should be less than {self.max_rigind}')
        
        self.train = self.df.iloc[0:(rigind+1)*increment, :]
        self.val = self.df.iloc[(rigind+1)*increment:(rigind+2)*increment, :]
        self.test = self.df.iloc[(rigind+2)*increment:(rigind+3)*increment, :]        

        self.train_x = self.train.drop(labels=['Rate of Penetration m/h'],axis=1)
        self.features_list = self.train_x.columns
        self.train_x = self.train_x.to_numpy()
        self.train_y = self.train['Rate of Penetration m/h'].to_numpy()

        self.val_x = self.val.drop(labels=['Rate of Penetration m/h'],axis=1).to_numpy()
        self.val_y = self.val['Rate of Penetration m/h'].to_numpy()

        self.test_x = self.test.drop(labels=['Rate of Penetration m/h'],axis=1).to_numpy()
        self.test_y = self.test['Rate of Penetration m/h'].to_numpy()

        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.scaler.fit(self.train_x)
        self.train_x = self.scaler.transform(self.train_x)
        self.val_x = self.scaler.transform(self.val_x)
        self.test_x = self.scaler.transform(self.test_x)

        if train_val_test == 'train':
            self.train_x = self.train_x
            self.train_y = self.train_y
        elif train_val_test == 'validation':
            self.train_x = self.val_x
            self.train_y = self.val_y
        elif train_val_test == 'test':
            self.train_x = self.test_x
            self.train_y = self.test_y
        else:
            raise ValueError('train_val_test should be one of [train, validation, test]')

    def __len__(self):
        return len(self.train_x)
    
    def __getitem__(self, idx):
        return torch.Tensor(self.train_x[idx, :]), torch.Tensor([self.train_y[idx]])
    


class ROPAllForOneDataset(Dataset):
    def __init__(
        self,
        val_well, 
        test_well,
        train_val_test: str,
        sample_interval: int = 1,
        ):
        # load_train_data
        
        data_folder = './dataset/processed_dataset/USROP/'        
        train_data_paths = [os.path.join(data_folder, well_path_dict[well]) for well in well_path_dict.keys() if well != val_well and well != test_well]
        pds_list = [pd.read_csv(path) for path in train_data_paths]
        self.train = pd.concat(pds_list, ignore_index=True)
        del self.train['Unnamed: 0']
        del self.train['Hole Depth (TVD) m']
        self.train_x = self.train.drop(labels=['Rate of Penetration m/h'],axis=1)
        self.features_list = self.train_x.columns
        self.train_x = self.train_x.to_numpy()
        self.train_y = self.train['Rate of Penetration m/h'].to_numpy()

        self.val = pd.read_csv(os.path.join(data_folder, well_path_dict[val_well]))
        del self.val['Unnamed: 0']
        del self.val['Hole Depth (TVD) m']
        self.val_x = self.val.drop(labels=['Rate of Penetration m/h'],axis=1).to_numpy()
        self.val_y = self.val['Rate of Penetration m/h'].to_numpy()


        self.test = pd.read_csv(os.path.join(data_folder, well_path_dict[test_well]))
        del self.test['Unnamed: 0']
        del self.test['Hole Depth (TVD) m']
        self.test_x = self.test.drop(labels=['Rate of Penetration m/h'],axis=1).to_numpy()
        self.test_y = self.test['Rate of Penetration m/h'].to_numpy()

        self.scaler = StandardScaler()
        self.scaler.fit(self.train_x)
        self.train_x = self.scaler.transform(self.train_x)
        self.val_x = self.scaler.transform(self.val_x)
        self.test_x = self.scaler.transform(self.test_x)

        if train_val_test == 'train':
            self.train_x = self.train_x
            self.train_y = self.train_y
        elif train_val_test == 'validation':
            self.train_x = self.val_x
            self.train_y = self.val_y
        elif train_val_test == 'test':
            self.train_x = self.test_x
            self.train_y = self.test_y
        else:
            raise ValueError('train_val_test should be one of [train, validation, test]')

    def __len__(self):
        return len(self.train_x)
    
    def __getitem__(self, idx):
        return torch.Tensor(self.train_x[idx, :]), torch.Tensor([self.train_y[idx]])






