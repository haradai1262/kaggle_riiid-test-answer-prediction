import sys
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.decomposition import TruncatedSVD

RANDOM_STATE = 46
INPUT_DIR = '../input'
FOLD_DIR = '../folds'
FOLD_NAME = 'vlatest_ALL_2p5M'

svd_dim = 3


def reduce_svd(features, n_components):
    dr_model = TruncatedSVD(n_components=n_components, random_state=46)
    features_dr = dr_model.fit_transform(features)
    return features_dr


if __name__ == "__main__":

    train = pd.read_feather(f'{INPUT_DIR}/train_v2.feather')
    folds = pd.read_feather(f'{FOLD_DIR}/train_folds_{FOLD_NAME}_v2.feather')
    question = pd.read_csv(f'{INPUT_DIR}/questions.csv')

    train = pd.merge(train, question[['question_id', 'part', 'tags']], left_on='content_id', right_on='question_id', how='left')
    train['part'] = train['part'].fillna(0.0).astype('int8')
    train['tags'] = train['tags'].fillna('0')

    train = train[train.row_id.isin(folds[folds.val == 0].row_id)]
    train = train[train.content_type_id == 0].reset_index(drop=True)
    train = train[['row_id', 'user_id', 'timestamp', 'content_type_id', 'question_id', 'part', 'answered_correctly']]

    i = 0
    rows = []
    for uid, user_df in tqdm(train.groupby('user_id')):
        user_df['prev_question_id_s1'] = user_df['question_id'].shift(1).fillna(-1).astype(int)
        user_df['timestamp_diff'] = user_df['timestamp'].diff() / 1e3 / 60
        rows.extend(user_df.values)

    content_lag_table = pd.DataFrame(rows, columns=[
        'row_id', 'user_id', 'timestamp', 'content_type_id',
        'question_id', 'part', 'answered_correctly', 'prev_question_id_s1', 'timestamp_diff'
    ])
    content_lag_table = content_lag_table.astype({
        'row_id': 'int',
        'user_id': 'int',
        'timestamp': 'int',
        'content_type_id': 'int',
        'row_id': 'int',
        'question_id': 'int',
        'part': 'int',
        'prev_question_id_s1': 'int',
    })

    q_num = question['question_id'].nunique()
    ans_trans_mat = np.zeros((q_num + 1, q_num + 1), dtype=np.int32)
    for i, j in tqdm(content_lag_table[['prev_question_id_s1', 'question_id']].values):
        if i == -1:
            i = q_num
        if j == -1:
            j = q_num
        ans_trans_mat[i, j] += 1
    ans_trans_mat = pd.DataFrame(ans_trans_mat)

    ans_corr_trans_mat = np.zeros((q_num + 1, q_num + 1), dtype=np.int32)
    for i, j in tqdm(content_lag_table[content_lag_table.answered_correctly == 1.0][['prev_question_id_s1', 'question_id']].values):
        if i == -1:
            i = q_num
        if j == -1:
            j = q_num
        ans_corr_trans_mat[i, j] += 1
    ans_corr_trans_mat = pd.DataFrame(ans_corr_trans_mat)

    mat = ans_trans_mat.values
    fpath = '../save/ans_trans_mat_edge_list.txt'
    with open(fpath, mode='w') as f:
        for i in tqdm(range(len(mat))):
            for j in range(len(mat)):
                if mat[i, j] == 0:
                    continue
                f.write(f'{i} {j} {mat[i,j]}\n')

    mat = ans_corr_trans_mat.values
    fpath = '../save/ans_corr_trans_mat_edge_list.txt'
    with open(fpath, mode='w') as f:
        for i in tqdm(range(len(mat))):
            for j in range(len(mat)):
                if mat[i, j] == 0:
                    continue
                f.write(f'{i} {j} {mat[i,j]}\n')

    features_dr = reduce_svd(ans_trans_mat.values, svd_dim)
    features_dr_T = reduce_svd(ans_trans_mat.values.T, svd_dim)

    g_features = pd.DataFrame([i for i in range(13523)], columns=['content_id'])
    g_features = pd.concat([
        g_features, pd.DataFrame(np.concatenate([features_dr[:-1, :], features_dr_T[:-1, :]], axis=1)).add_prefix('gsvd_')
    ], axis=1)
    g_features.to_feather(f'../save/ans_trans_Graph_SVD_svd{svd_dim}.feather')

    features_dr = reduce_svd(ans_corr_trans_mat.values, svd_dim)
    features_dr_T = reduce_svd(ans_corr_trans_mat.values.T, svd_dim)

    g_features = pd.DataFrame([i for i in range(13523)], columns=['content_id'])
    g_features = pd.concat([
        g_features, pd.DataFrame(np.concatenate([features_dr[:-1, :], features_dr_T[:-1, :]], axis=1)).add_prefix('corr_gsvd_')
    ], axis=1)
    g_features.to_feather(f'../save/ans_corr_trans_Graph_SVD_svd{svd_dim}.feather')
