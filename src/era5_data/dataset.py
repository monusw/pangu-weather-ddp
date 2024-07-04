import os
from typing import *

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset

class Era5Dataset(Dataset):
    r""" ERA5 dataset (NetCDF format) wrapper for PyTorch DataLoader.
    Args:
        data_path (str): Directory containing the ERA5 NetCDF files.
            The data should be organized in the following structure:
            data_path
            ├── surface
            │   ├── 20070101.nc
            │   └── ...
            └── upper
                ├── 20070101.nc
                └── ...
            Each NetCDF file contains all the data for a single day. (24 hours)
        begin_date (str): Start date of the dataset in the format 'YYYYMMDD'. (e.g. '20070101')
        end_date (str): End date of the dataset in the format 'YYYYMMDD'. (e.g. '20070102')
        freq (str): Frequency of the data. Format: 'nh', n can be 1-24. Default is '1h'.
        lead_time (int): Lead time of the forecast in hours. Default is 1.
        normalize_data_path (str): .npz file containing the mean and standard deviation of the data.
            keys: 'upper_mean', 'upper_std',    (5, 13, 721, 1440)
                  'surface_mean', 'surface_std' (4, 721, 1440)
            If None, we do not normalize the data. Default is None.

    Note: `begin_date` is inclusive, `end_date` is exclusive. For example, if begin_date='20070101'
        and end_date='20070102', freq='1h', the original dataset will contain data
        from 20070101 00:00 to 20070101 23:00. (24 data points)
        Take `lead_time` into consideration, the final dataset will contain (24 - `lead_time`) data points.
    """
    def __init__(self,
                 data_path,
                 begin_date,
                 end_date,
                 freq='1h',
                 lead_time=1,
                 normalize_data_path=None,
                 ):
        self.data_path = data_path
        self.normalize = (normalize_data_path is not None)
        self.normalize_data_path = normalize_data_path

        self.data_keys = list(pd.date_range(start=begin_date, end=end_date, freq=freq, inclusive='left'))
        # ground truth data keys
        gt_begin_date = pd.to_datetime(begin_date) + pd.DateOffset(hours=lead_time)
        gt_end_date = pd.to_datetime(end_date) + pd.DateOffset(hours=lead_time)
        self.gt_data_keys = list(pd.date_range(start=gt_begin_date, end=gt_end_date, freq=freq, inclusive='left'))

        assert len(self.data_keys) == len(self.gt_data_keys)

        final_i = 0
        for i in range(len(self.data_keys) - 1, -1, -1):
            final_i = i
            if self.gt_data_keys[i] < pd.to_datetime(end_date):
                final_i += 1
                break
        assert final_i > 0
        self.data_keys = self.data_keys[:final_i]
        self.gt_data_keys = self.gt_data_keys[:final_i]

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, index):
        # TODO
        pass

    def __repr__(self):
        return self.__class__.__name__

    def _load_data(self, index: int):
        ts = self.data_keys[index]
        gt_ts = self.gt_data_keys[index]

        # foramat to YYYYMMDD
        ts_str = ts.strftime('%Y%m%d')
        gt_ts_str = gt_ts.strftime('%Y%m%d')
        # TODO



    def LoadData(self, key):
        """
        Input
            key: datetime object, input time
        Return
            input: numpy
            input_surface: numpy
            target: numpy label
            target_surface: numpy label
            (start_time_str, end_time_str): string, datetime(target time - input time) = horizon
        """
        # start_time datetime obj
        start_time = key
        # convert datetime obj to string for matching file name and return key
        start_time_str = datetime.strftime(key, '%Y%m%d%H')

        # target time = start time + horizon
        end_time = key + timedelta(hours=self.horizon)
        end_time_str = end_time.strftime('%Y%m%d%H')

        # Prepare the input_surface dataset
        # print(start_time_str[0:6])
        input_surface_dataset = xr.open_dataset(
            os.path.join(self.nc_path, 'surface', 'surface_{}.nc'.format(start_time_str[0:6])))  # 201501
        if 'expver' in input_surface_dataset.keys():
            input_surface_dataset = input_surface_dataset.sel(time=start_time, expver=5)
        else:
            input_surface_dataset = input_surface_dataset.sel(time=start_time)

        # Prepare the input_upper dataset
        input_upper_dataset = xr.open_dataset(
            os.path.join(self.nc_path, 'upper', 'upper_{}.nc'.format(start_time_str[0:8])))
        if 'expver' in input_upper_dataset.keys():
            input_upper_dataset = input_upper_dataset.sel(time=start_time, expver=5)
        else:
            input_upper_dataset = input_upper_dataset.sel(time=start_time)
        # make sure upper and surface variables are at the same time
        assert input_surface_dataset['time'] == input_upper_dataset['time']
        # input dataset to input numpy
        input, input_surface = self.nctonumpy(input_upper_dataset, input_surface_dataset)

        # Prepare the target_surface dataset
        target_surface_dataset = xr.open_dataset(
            os.path.join(self.nc_path, 'surface', 'surface_{}.nc'.format(end_time_str[0:6])))  # 201501
        if 'expver' in input_surface_dataset.keys():
            target_surface_dataset = target_surface_dataset.sel(time=end_time, expver=5)
        else:
            target_surface_dataset = target_surface_dataset.sel(time=end_time)
        # Prepare the target upper dataset
        target_upper_dataset = xr.open_dataset(
            os.path.join(self.nc_path, 'upper', 'upper_{}.nc'.format(end_time_str[0:8])))
        if 'expver' in target_upper_dataset.keys():
            target_upper_dataset = target_upper_dataset.sel(time=end_time, expver=5)
        else:
            target_upper_dataset = target_upper_dataset.sel(time=end_time)
        # make sure the target upper and surface variables are at the same time
        assert target_upper_dataset['time'] == target_surface_dataset['time']
        # target dataset to target numpy
        target, target_surface = self.nctonumpy(target_upper_dataset, target_surface_dataset)

        return input, input_surface, target, target_surface, (start_time_str, end_time_str)


def era5_nctonumpy(dataset_upper: xr.Dataset, dataset_surface: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    upper_z = dataset_upper['z'].values.astype(np.float32)  # (13, 721, 1440)
    upper_q = dataset_upper['q'].values.astype(np.float32)
    upper_t = dataset_upper['t'].values.astype(np.float32)
    upper_u = dataset_upper['u'].values.astype(np.float32)
    upper_v = dataset_upper['v'].values.astype(np.float32)
    upper = np.concatenate((upper_z[np.newaxis, ...], upper_q[np.newaxis, ...], upper_t[np.newaxis, ...],
                            upper_u[np.newaxis, ...], upper_v[np.newaxis, ...]), axis=0)
    assert upper.shape == (5, 13, 721, 1440)
    # levels in descending order, require new memery space
    upper = upper[:, ::-1, :, :].copy()

    surface_mslp = dataset_surface['msl'].values.astype(np.float32)  # (721,1440)
    surface_u10 = dataset_surface['u10'].values.astype(np.float32)
    surface_v10 = dataset_surface['v10'].values.astype(np.float32)
    surface_t2m = dataset_surface['t2m'].values.astype(np.float32)
    surface = np.concatenate((surface_mslp[np.newaxis, ...], surface_u10[np.newaxis, ...],
                                surface_v10[np.newaxis, ...], surface_t2m[np.newaxis, ...]), axis=0)
    assert surface.shape == (4, 721, 1440)

    return upper, surface

