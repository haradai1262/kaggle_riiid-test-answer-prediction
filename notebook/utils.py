import pandas as pd
import numpy as np
import random
import os
import json
import time
import yaml
from contextlib import contextmanager
# import torch
import logging
import cloudpickle
import joblib

from easydict import EasyDict as edict
from collections import OrderedDict
from abc import ABCMeta, abstractmethod


def load_from_pkl(load_path):
    frb = open(load_path, 'rb')
    obj = cloudpickle.loads(frb.read())
    return obj


def save_as_pkl(obj, save_path):
    fwb = open(save_path, 'wb')
    fwb.write(cloudpickle.dumps(obj))
    return


def seed_everything(seed=46):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for i, col in enumerate(df.columns):
        try:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        except ValueError:
            continue

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


class Timer:
    def __init__(self):
        self.processing_time = 0

    @contextmanager
    def timer(self, name):
        logging.info(f'[{name}] start')
        t0 = time.time()
        yield
        t1 = time.time()
        processing_time = t1 - t0
        self.processing_time += round(processing_time, 2)
        if self.processing_time < 60:
            logging.info(f'[{name}] done in {processing_time:.0f} s (Total: {self.processing_time:.2f} sec)')
        elif self.processing_time < 3600:
            logging.info(f'[{name}] done in {processing_time:.0f} s (Total: {self.processing_time / 60:.2f} min)')
        else:
            logging.info(f'[{name}] done in {processing_time:.0f} s (Total: {self.processing_time / 3600:.2f} hour)')

    def get_processing_time(self):
        return round(self.processing_time, 2)


# =============================================================================
# Data Processor
# =============================================================================
class DataProcessor(metaclass=ABCMeta):

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def save(self, path, data):
        pass


class YmlPrrocessor(DataProcessor):

    def load(self, path):
        with open(path, 'r') as yf:
            yaml_file = yaml.load(yf, Loader=yaml.SafeLoader)
        return edict(yaml_file)

    def save(self, path, data):
        def represent_odict(dumper, instance):
            return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())

        yaml.add_representer(OrderedDict, represent_odict)
        yaml.add_representer(edict, represent_odict)

        with open(path, 'w') as yf:
            yf.write(yaml.dump(OrderedDict(data), default_flow_style=False))


class CsvProcessor(DataProcessor):

    def __init__(self, sep):
        self.sep = sep

    def load(self, path, sep=','):
        data = pd.read_csv(path, sep=sep)
        return data

    def save(self, path, data):
        data.to_csv(path, index=False)


class FeatherProcessor(DataProcessor):

    def load(self, path):
        data = pd.read_feather(path)
        return data

    def save(self, path, data):
        data.to_feather(path)


class PickleProcessor(DataProcessor):

    def load(self, path):
        data = joblib.load(path)
        return data

    def save(self, path, data):
        joblib.dump(data, path, compress=True)


class NpyProcessor(DataProcessor):

    def load(self, path):
        data = np.load(path)
        return data

    def save(self, path, data):
        np.save(path, data)


class JsonProcessor(DataProcessor):

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
        return data

    def save(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)


class DataHandler:
    def __init__(self):
        self.data_encoder = {
            '.yml': YmlPrrocessor(),
            '.csv': CsvProcessor(sep=','),
            '.tsv': CsvProcessor(sep='\t'),
            '.feather': FeatherProcessor(),
            '.pkl': PickleProcessor(),
            '.npy': NpyProcessor(),
            '.json': JsonProcessor(),
        }

    def load(self, path):
        extension = self._extract_extension(path)
        data = self.data_encoder[extension].load(path)
        return data

    def save(self, path, data):
        extension = self._extract_extension(path)
        self.data_encoder[extension].save(path, data)

    def _extract_extension(self, path):
        return os.path.splitext(path)[1]