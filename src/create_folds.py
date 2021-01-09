import pandas as pd
import random

from utils import (
    seed_everything,
    Timer
)


# RANDOM_STATE = 46
RANDOM_STATE = 20201209

INPUT_DIR = '../input'
FOLD_DIR = '../folds'

# FOLD_NAME = 'vlatest_ALL_2p5M'
FOLD_NAME = 'vlatest_10M_2p5M'


def rand_time(max_time_stamp):
    interval = MAX_TIME_STAMP - max_time_stamp
    rand_time_stamp = random.randint(0, interval)
    if rand_time_stamp > MAX_TIME_STAMP - max_time_stamp:
        rand_time_stamp = MAX_TIME_STAMP - max_time_stamp
    return rand_time_stamp


if __name__ == "__main__":

    t = Timer()
    with t.timer(f'fix seed RANDOM_STATE:{RANDOM_STATE}'):
        seed_everything(RANDOM_STATE)

    # train = pd.read_feather(f'{INPUT_DIR}/train.feather')
    train = pd.read_feather(f'{INPUT_DIR}/train_v2.feather')

    # https://www.kaggle.com/its7171/cv-strategy
    if FOLD_NAME == 'vlatest_10M_2p5M':

        max_timestamp_u = train[['user_id', 'timestamp']].groupby(['user_id']).agg(['max']).reset_index()
        max_timestamp_u.columns = ['user_id', 'max_time_stamp']
        MAX_TIME_STAMP = max_timestamp_u.max_time_stamp.max()

        max_timestamp_u['rand_time_stamp'] = max_timestamp_u.max_time_stamp.apply(rand_time)
        train = train.merge(max_timestamp_u, on='user_id', how='left')
        train['viretual_time_stamp'] = train.timestamp + train['rand_time_stamp']

        train = train.sort_values(['viretual_time_stamp', 'row_id'])

        tra_size = 10000000
        val_size = 2500000
        tra_idx = train[-tra_size - val_size:-val_size].index
        val_idx = train[-val_size:].index

        train['val'] = -1
        train.loc[tra_idx, 'val'] = 0
        train.loc[val_idx, 'val'] = 1

        print(train.val.value_counts())
        print(f'tra_target_mean: {train[(train.val==0) & (train.content_type_id == 0)].answered_correctly.mean()}')
        print(f'val_target_mean: {train[(train.val==1) & (train.content_type_id == 0)].answered_correctly.mean()}')
        train[['row_id', 'val']].reset_index().to_feather(f'{FOLD_DIR}/train_folds_{FOLD_NAME}_v2_{RANDOM_STATE}.feather')

    if FOLD_NAME == 'vlatest_ALL_2p5M':

        max_timestamp_u = train[['user_id', 'timestamp']].groupby(['user_id']).agg(['max']).reset_index()
        max_timestamp_u.columns = ['user_id', 'max_time_stamp']
        MAX_TIME_STAMP = max_timestamp_u.max_time_stamp.max()

        max_timestamp_u['rand_time_stamp'] = max_timestamp_u.max_time_stamp.apply(rand_time)
        train = train.merge(max_timestamp_u, on='user_id', how='left')
        train['viretual_time_stamp'] = train.timestamp + train['rand_time_stamp']

        train = train.sort_values(['viretual_time_stamp', 'row_id'])

        tra_size = 100000000
        val_size = 2500000
        tra_idx = train[-tra_size - val_size:-val_size].index
        val_idx = train[-val_size:].index

        train['val'] = -1
        train.loc[tra_idx, 'val'] = 0
        train.loc[val_idx, 'val'] = 1

        print(train.val.value_counts())
        print(f'tra_target_mean: {train[(train.val==0) & (train.content_type_id == 0)].answered_correctly.mean()}')
        print(f'val_target_mean: {train[(train.val==1) & (train.content_type_id == 0)].answered_correctly.mean()}')
        train[['row_id', 'val']].reset_index().to_feather(f'{FOLD_DIR}/train_folds_{FOLD_NAME}_v2_{RANDOM_STATE}.feather')