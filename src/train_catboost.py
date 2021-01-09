
import os
import sys
import shutil
import gc
import logging
import pandas as pd
import numpy as np
from collections import defaultdict
from bitarray import bitarray
import pickle5 
from tqdm import tqdm

from collections import Counter
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from scipy.stats import multinomial
import statistics
from gensim.models import Word2Vec

import mlflow

from sklearn.metrics import roc_auc_score

from utils import (
    seed_everything,
    Timer,
    reduce_mem_usage,
    save_as_pkl
)
from preprocess_func import (
    prep_tags,
    part2lr,
    merge_questions_w_prep,
    prep_base,
    add_modified_target_based_on_user_answer,
    make_repeat_table,
    make_elapsed_time_table
)

sys.path.append('/home/myaun/catboost/catboost/python-package')
import catboost
print('CAT Version', catboost.__version__)
print('Python     : ' + sys.version.split('\n')[0])
print('Numpy      : ' + np.__version__)
print('Pandas     : ' + pd.__version__)

INPUT_DIR = '../input'
FOLD_DIR = '../folds'
EXP_DIR = '../exp'
EXP_CONFIG = '001'
sys.path.append(f'{EXP_DIR}/{EXP_CONFIG}')
import config

RANDOM_STATE = config.RANDOM_STATE
FOLD_NAME = config.FOLD_NAME

use_features = config.use_features
print('len(use_features)', len(use_features))
FE_MODEL_PARAMS = config.FE_MODEL_PARAMS
CAT_PARAMS = config.CAT_PARAMS

content_tsne_path = config.content_tsne_path
content_tsne_kmeans_path = config.content_tsne_kmeans_path
content_cow2v_kmeans_path = config.content_cow2v_kmeans_path
w2v_feature_path = config.w2v_feature_path
g_feature_path = config.g_feature_path
corr_g_feature_path = config.corr_g_feature_path
ge_dw_feature_path = config.ge_dw_feature_path
ge_s2v_feature_path = config.ge_s2v_feature_path
content_graph_metric_features = config.content_graph_metric_features

SAVE_EXTACTED_FEATURES = True

EXP_NAME = f'{FOLD_NAME}__CAT'
if not os.path.exists(f'../features/{EXP_NAME}/'):
    os.mkdir(f'../features/{EXP_NAME}/')


def save_mlflow(run_id, score):

    mlflow.log_param("RANDOM_STATE", RANDOM_STATE)
    mlflow.log_param("FOLD_NAME", FOLD_NAME)

    for feat in use_features:
        mlflow.log_param(f'f__{feat}', 1)

    mlflow.log_param('p__TE_smooth', FE_MODEL_PARAMS['TE_smooth'])
    mlflow.log_param('p__ansrec_max_len', FE_MODEL_PARAMS['ansrec_max_len'])
    mlflow.log_param('p__timestamprec_max_len', FE_MODEL_PARAMS['timestamprec_max_len'])
    mlflow.log_param('p__prev_question_len', FE_MODEL_PARAMS['prev_question_len'])
    mlflow.log_param('p__use_previous_similarities_num', FE_MODEL_PARAMS['use_previous_similarities_num'])

    mlflow.log_param('p__CAT_depth', CAT_PARAMS['depth'])
    mlflow.log_param('p__CAT_learning_rate', CAT_PARAMS['learning_rate'])
    mlflow.log_param('p__CAT_bagging_temperature', CAT_PARAMS['bagging_temperature'])

    mlflow.log_metric("score", score)
    return


class riiidFE:

    def __init__(self, model_parames):
        self.n_skill = 13523
        self.n_part = 7

        self.user_corr_cnt_dict = defaultdict(int)
        self.user_cnt_dict = defaultdict(int)

        self.user_answer_u_dict = defaultdict(lambda: np.zeros(4, dtype=np.uint16))
        self.incorr_user_answer_u_dict = defaultdict(lambda: np.zeros(4, dtype=np.uint16))

        self.user_first_content_dict = defaultdict(lambda: -1)
        self.user_first10_corr_rate_dict = defaultdict(float)
        self.user_first1day_corr_rate_dict = defaultdict(float)
        self.user_first1week_corr_rate_dict = defaultdict(float)
        self.user_first1day_cnt_dict = defaultdict(int)
        self.user_first1week_cnt_dict = defaultdict(int)

        self.user_corr_cnt_in_session_short_dict = defaultdict(int)
        self.user_cnt_in_session_short_dict = defaultdict(int)

        self.user_session_short_cnt_dict = defaultdict(int)
        self.user_prev_session_short_ts_dict = defaultdict(int)

        self.ques_kmeans100_u_dict = defaultdict(lambda: np.zeros(100, dtype=np.uint16))
        self.ques_corr_kmeans100_u_dict = defaultdict(lambda: np.zeros(100, dtype=np.uint16))
        self.ques_co_kmeans100_u_dict = defaultdict(lambda: np.zeros(100, dtype=np.uint16))
        self.ques_corr_co_kmeans100_u_dict = defaultdict(lambda: np.zeros(100, dtype=np.uint16))
        self.ques_inco_kmeans100_u_dict = defaultdict(lambda: np.zeros(100, dtype=np.uint16))
        self.ques_corr_inco_kmeans100_u_dict = defaultdict(lambda: np.zeros(100, dtype=np.uint16))

        self.part_u_dict = defaultdict(lambda: np.zeros(self.n_part, dtype=np.uint16))
        self.part_corr_u_dict = defaultdict(lambda: np.zeros(self.n_part, dtype=np.uint16))
        self.tag_u_dict = defaultdict(lambda: np.zeros(189, dtype=np.uint16))
        self.tag_corr_u_dict = defaultdict(lambda: np.zeros(189, dtype=np.uint16))

        self.lecture_latest_u_dict = defaultdict(lambda: -1)
        self.lecture_type_cnt_u_dict = defaultdict(lambda: np.zeros(4, dtype=np.uint16))
        self.lecture_part_cnt_u_dict = defaultdict(lambda: np.zeros(8, dtype=np.uint16))
        self.ans_num_from_latest_lecture_u_dict = defaultdict(lambda: -1)

        self.hadexp_sum_u_dict = defaultdict(int)
        self.hadexp_cnt_u_dict = defaultdict(int)

        self.user_rec_dict = defaultdict(lambda: bitarray(0, endian='little'))
        self.user_prev_ques_dict = defaultdict(list)

        self.user_timestamp_dict = defaultdict(list)
        self.user_timestamp_incorr_dict = defaultdict(int)
        self.user_et_dict = defaultdict(list)

        self.user_et_sum_dict = defaultdict(int)

        self.session_short_th = 600000  # 30min model_parames['session_th']

        self.user_latest_tci_dict = defaultdict(lambda: -1)

        self.smooth = model_parames['TE_smooth']
        self.ansrec_max_len = model_parames['ansrec_max_len']
        self.timestamprec_max_len = model_parames['timestamprec_max_len']
        self.prev_question_len = model_parames['prev_question_len']
        self.use_previous_similarities_num = model_parames['use_previous_similarities_num']

        self.loop_features = ['timestamp', 'content_type_id', 'user_id', 'content_id', 'part', 'tags',
                              'prior_question_had_explanation', 'prior_question_elapsed_time', 'task_container_id']
        self.use_labels = ['answered_correctly', 'user_answer']

        self.se_smooth = model_parames['sequence_te_smooth']

        tmp = pd.read_feather('../save/u_timestamp_lag_median.feather')
        tmp['timestamp_lag_median'].mean()
        self.user_difftime_median = {int(i): j for i, j in tmp.values}
        self.user_difftime_median['all'] = tmp['timestamp_lag_median'].mean()

        self.lecture_id2part = {i: j - 1 for i, j in lecture[['lecture_id', 'part']].values}
        self.lecture_id2type = {i: j for i, j in lecture[['lecture_id', 'type_of']].values}

        self.content_id2corrans = {i: j for i, j in question[['question_id', 'correct_answer']].values}

        content_clusters = pd.read_feather(content_tsne_kmeans_path)
        self.content_id2kmean100 = {i: j for i, j in content_clusters[['content_id', 'kmeans_100']].values}

        content_clusters = pd.read_feather(content_cow2v_kmeans_path)
        self.content_id2cokmean100 = {i: j for i, j in content_clusters[['content_id', 'seqs_cluster_100']].values}
        self.content_id2incokmean100 = {i: j for i, j in content_clusters[['content_id', 'incorr_seqs_cluster_100']].values}

        with open('../save/Word2vec_simmat_withlec_nf20_sg1_win50_minc1.pkl', mode='rb') as f:
            self.w2v_simmat = pickle5.load(f)
        with open('../save/Word2vec_cname2idx_withlec_nf20_sg1_win50_minc1.pkl', mode='rb') as f:
            self.w2v_cname2idx = pickle5.load(f)

    def set_use_features(self, use_features):
        self.use_features = use_features

    def set_train_mn(self, train):
        self.train_mn = train['answered_correctly'].mean()
        self.train_wans_mn = train['weighted_answered_correctly'].mean()
        self.train_ws_mn = train['weighted_score'].mean()
        print(self.train_mn, self.train_wans_mn, self.train_ws_mn)

    def set_repeat_mn(self, repeat):
        self.repeat_mn = repeat['answered_correctly'].mean()
        self.repeat_wans_mn = repeat['weighted_answered_correctly'].mean()
        self.repeat_ws_mn = repeat['weighted_score'].mean()
        print(self.repeat_mn, self.repeat_wans_mn, self.repeat_ws_mn)

    def set_cat_te_dict(self, X_tra_wo_lec, question):
        col = 'part'
        te_feat = self.fe_te_sm(X_tra_wo_lec, [col], target='answered_correctly', mn=self.train_mn)
        self.part_te_dict = {int(i): j for i, j in te_feat.values}

        col = 'content_id'
        te_feat = self.fe_te_sm(X_tra_wo_lec, [col], target='answered_correctly', mn=self.train_mn)
        self.ques_te_dict = {int(i): j for i, j in te_feat.values}
        for cid in question['question_id'].unique():
            if cid not in self.ques_te_dict:
                self.ques_te_dict[int(cid)] = self.train_mn

        self.question2part = {i[0]: i[1] for i in question[['question_id', 'part']].values}
        self.question2part[self.n_skill + 1] = self.n_part + 1

    def update_user_log_dict(self, uid, cid, part, tags, ans, uans, timestamp, et, hexp, dt, lt, tci):

        if self.user_cnt_dict[uid] == 0:
            self.user_first_content_dict[uid] = cid

        if self.user_cnt_dict[uid] >= 1 and self.user_cnt_dict[uid] <= 10:
            self.user_first10_corr_rate_dict[uid] = sum(self.user_rec_dict[uid]) / float(len(self.user_rec_dict[uid]) + 1)

        if timestamp <= 86400000:
            self.user_first1day_cnt_dict[uid] = self.user_cnt_dict[uid]
            self.user_first1day_corr_rate_dict[uid] = sum(self.user_rec_dict[uid]) / float(len(self.user_rec_dict[uid]) + 1)
        if timestamp <= 604800000:  # 86400000*7
            self.user_first1week_cnt_dict[uid] = self.user_cnt_dict[uid]
            self.user_first1week_corr_rate_dict[uid] = sum(self.user_rec_dict[uid]) / float(len(self.user_rec_dict[uid]) + 1)

        self.user_corr_cnt_dict[uid] += ans
        self.user_cnt_dict[uid] += 1
        if self.ans_num_from_latest_lecture_u_dict[uid] != -1:
            self.ans_num_from_latest_lecture_u_dict[uid] += 1

        self.user_corr_cnt_in_session_short_dict[uid] += ans

        self.user_cnt_in_session_short_dict[uid] += 1

        if len(self.user_rec_dict[uid]) == self.ansrec_max_len:
            self.user_rec_dict[uid].pop(0)
        self.user_rec_dict[uid].append(ans)

        if len(self.user_prev_ques_dict[uid]) == self.prev_question_len:
            self.user_prev_ques_dict[uid].pop(0)
        self.user_prev_ques_dict[uid].append(cid)

        if hexp == 1:
            self.hadexp_sum_u_dict[uid] += ans
            self.hadexp_cnt_u_dict[uid] += 1

        self.ques_kmeans100_u_dict[uid][self.content_id2kmean100[cid]] += 1
        self.ques_co_kmeans100_u_dict[uid][self.content_id2cokmean100[cid]] += 1
        self.ques_inco_kmeans100_u_dict[uid][self.content_id2incokmean100[cid]] += 1

        self.part_u_dict[uid][part - 1] += 1
        for t in tags:
            self.tag_u_dict[uid][t] += 1

        self.user_answer_u_dict[uid][uans] += 1
        if ans == 1:
            self.ques_corr_kmeans100_u_dict[uid][self.content_id2kmean100[cid]] += 1
            self.ques_corr_co_kmeans100_u_dict[uid][self.content_id2cokmean100[cid]] += 1
            self.ques_corr_inco_kmeans100_u_dict[uid][self.content_id2incokmean100[cid]] += 1

            self.part_corr_u_dict[uid][part - 1] += 1
            for t in tags:
                self.tag_corr_u_dict[uid][t] += 1
        else:  # incorrect
            self.user_timestamp_incorr_dict[uid] = timestamp
            self.incorr_user_answer_u_dict[uid][uans] += 1

        if dt > self.session_short_th:
            self.user_session_short_cnt_dict[uid] += 1

        if len(self.user_timestamp_dict[uid]) == self.timestamprec_max_len:
            self.user_timestamp_dict[uid].pop(0)
        self.user_timestamp_dict[uid].append(timestamp)

        if len(self.user_et_dict[uid]) == self.timestamprec_max_len:
            self.user_et_dict[uid].pop(0)
        self.user_et_dict[uid].append(et)
        self.user_et_sum_dict[uid] += et

        self.user_latest_tci_dict[uid] = tci
        return

    def dataframe_process(self, user_feats_df):
        user_feats_df['u_corr_rate'] = user_feats_df['u_corr_cnt'] / user_feats_df['u_cnt']
        user_feats_df['u_hade_corr_rate'] = user_feats_df['u_hade_corr_cnt'] / user_feats_df['u_hade_cnt']
        user_feats_df['u_hade_rate'] = user_feats_df['u_hade_cnt'] / user_feats_df['u_cnt']
        user_feats_df['u_hade_div_corr_cnt'] = user_feats_df['u_hade_cnt'] / (user_feats_df['u_corr_cnt'] + 1)

        user_feats_df['u_hist_queskm100_corr_rate'] = user_feats_df['u_hist_queskm100_corr_num'] / (user_feats_df['u_hist_queskm100_num'] + 1)
        user_feats_df['u_hist_ques_cokm100_corr_rate'] = user_feats_df['u_hist_ques_cokm100_corr_num'] / (user_feats_df['u_hist_ques_cokm100_num'] + 1)
        user_feats_df['u_hist_ques_incokm100_corr_rate'] = user_feats_df['u_hist_ques_incokm100_corr_num'] / (user_feats_df['u_hist_ques_incokm100_num'] + 1)

        user_feats_df['u_corr_rate_smooth'] = (user_feats_df['u_cnt'] * user_feats_df['u_corr_rate'] + self.smooth * self.train_mn) / (user_feats_df['u_cnt'] + self.smooth)

        user_feats_df['u_hist_part_num_corr_rate'] = user_feats_df['u_hist_part_corr_num'] / user_feats_df['u_hist_part_num']

        user_feats_df['u_hist_tag_num_corr_rate'] = user_feats_df['u_hist_tag_corr_num'] / user_feats_df['u_hist_tag_num']

        user_feats_df['u_prev_diff_lag_time'] = user_feats_df['u_prev_difftime'] / (user_feats_df['u_prev_lagtime'] + 1)
        user_feats_df['u_prev_lag_diff_time'] = user_feats_df['u_prev_lagtime'] / (user_feats_df['u_prev_difftime'] + 1)
        return user_feats_df

    def seq2dec_feat(self, user_records, w_size):
        seq2dec_feats = 0.
        for i in range(w_size):
            seq2dec_feats += user_records[-(i + 1)] * (10 ** -i)
        return seq2dec_feats

    def add_user_feats(self, df, add_feat=True, update_dict=True, val=False):

        def add_user_latest_record_feat(cnt, user_records):
            n_len = len(user_records)
            if n_len == 0:
                u_latest10_corr_rate[cnt] = np.nan
                u_latest30_corr_rate[cnt] = np.nan
                u_in_seq_corr_rate[cnt] = np.nan
            else:
                u_latest10_corr_rate[cnt] = sum(user_records[-10:]) / len(user_records[-10:])
                u_latest30_corr_rate[cnt] = sum(user_records[-30:]) / len(user_records[-30:])
                u_in_seq_corr_rate[cnt] = sum(user_records) / len(user_records)

                corr_rate = self.user_corr_cnt_dict[uid] / self.user_cnt_dict[uid]

                u_latest10_corr_rate_diff_all[cnt] = u_latest10_corr_rate[cnt] - corr_rate
                u_latest30_corr_rate_diff_all[cnt] = u_latest30_corr_rate[cnt] - corr_rate

            if n_len > 3:
                seq2dec_w3[cnt] = self.seq2dec_feat(user_records, w_size=3)
            if n_len > 7:
                seq2dec_w7[cnt] = self.seq2dec_feat(user_records, w_size=7)

            return

        def w2v_sim(i, j):
            if i > j:
                sim = self.w2v_simmat[j, i]
            else:
                sim = self.w2v_simmat[i, j]
            return sim

        def cal_sims_x__previous_xxx(mark_x, mark_xxx, x, xxx):
            sims = []
            for i in range(1, len(xxx) + 1):
                sims.append(w2v_sim(self.w2v_cname2idx[f'{mark_x}{x}'], self.w2v_cname2idx[f'{mark_xxx}{xxx[-1*i]}']))
            return sims

        def add_user_ques_record_feat(cnt, uid, cid, part, tags, val=False):

            if cid in self.user_prev_ques_dict[uid]:
                u_hist_ques_in_seq[cnt] = 1
                u_hist_ques_in_seq_num[cnt] = self.user_prev_ques_dict[uid].count(cid)

            u_hist_queskm100_num[cnt] = self.ques_kmeans100_u_dict[uid][self.content_id2kmean100[cid]]
            u_hist_queskm100_corr_num[cnt] = self.ques_corr_kmeans100_u_dict[uid][self.content_id2kmean100[cid]]

            u_hist_ques_cokm100_num[cnt] = self.ques_co_kmeans100_u_dict[uid][self.content_id2cokmean100[cid]]
            u_hist_ques_cokm100_corr_num[cnt] = self.ques_corr_co_kmeans100_u_dict[uid][self.content_id2cokmean100[cid]]

            u_hist_ques_incokm100_num[cnt] = self.ques_inco_kmeans100_u_dict[uid][self.content_id2incokmean100[cid]]
            u_hist_ques_incokm100_corr_num[cnt] = self.ques_corr_inco_kmeans100_u_dict[uid][self.content_id2incokmean100[cid]]

            user_prev_questions = self.user_prev_ques_dict[uid][-1 * self.use_previous_similarities_num:]

            prev_w2v_qq_sims = cal_sims_x__previous_xxx('c', 'c', x=cid, xxx=user_prev_questions)
            prev_w2v_pq_sims = cal_sims_x__previous_xxx('p', 'c', x=part, xxx=user_prev_questions)

            prev_w2v_qq_sims_incorr = []
            prev_w2v_pq_sims_incorr = []
            for i in range(len(user_prev_questions)):
                if self.user_rec_dict[uid][-1 * (i + 1)] == 0:
                    prev_w2v_qq_sims_incorr.append(prev_w2v_qq_sims[i])
                    prev_w2v_pq_sims_incorr.append(prev_w2v_pq_sims[i])

            if len(user_prev_questions) >= 1:
                prev_idx = 1
                ans_flag = self.user_rec_dict[uid][-1 * prev_idx]
                if ans_flag == 0:
                    ans_flag = -1
                u_hist_ques_sim_prevques[cnt] = prev_w2v_qq_sims[prev_idx - 1]
                u_hist_ques_sim_prevques_ans[cnt] = prev_w2v_qq_sims[prev_idx - 1] * ans_flag
                prev_qp_sim = w2v_sim(self.w2v_cname2idx[f'c{cid}'], self.w2v_cname2idx[f'p{self.question2part[user_prev_questions[-1]]}'])
                u_hist_ques_sim_prevpart[cnt] = prev_qp_sim  # prev_w2v_qp_sims[prev_idx - 1]
                u_hist_ques_sim_prevpart_ans[cnt] = prev_qp_sim * ans_flag  # prev_w2v_qp_sims[prev_idx - 1] * ans_flag
                u_hist_part_sim_prevques[cnt] = prev_w2v_pq_sims[prev_idx - 1]
                u_hist_part_sim_prevques_ans[cnt] = prev_w2v_pq_sims[prev_idx - 1] * ans_flag

                u_hist_ques_sim_prevques_mean[cnt] = sum(prev_w2v_qq_sims) / len(prev_w2v_qq_sims)
                u_hist_ques_sim_prevques_max[cnt] = max(prev_w2v_qq_sims)
                u_hist_ques_sim_prevques_min[cnt] = min(prev_w2v_qq_sims)

                u_hist_part_sim_prevques_mean[cnt] = sum(prev_w2v_pq_sims) / len(prev_w2v_pq_sims)
                u_hist_part_sim_prevques_max[cnt] = max(prev_w2v_pq_sims)
                u_hist_part_sim_prevques_min[cnt] = min(prev_w2v_pq_sims)

            if len(prev_w2v_qq_sims_incorr) >= 1:
                u_hist_ques_sim_prevques_incorr_mean[cnt] = sum(prev_w2v_qq_sims_incorr) / len(prev_w2v_qq_sims_incorr)
                u_hist_ques_sim_prevques_incorr_max[cnt] = max(prev_w2v_qq_sims_incorr)
                u_hist_ques_sim_prevques_incorr_min[cnt] = min(prev_w2v_qq_sims_incorr)

                u_hist_part_sim_prevques_incorr_mean[cnt] = sum(prev_w2v_pq_sims_incorr) / len(prev_w2v_pq_sims_incorr)
                u_hist_part_sim_prevques_incorr_max[cnt] = max(prev_w2v_pq_sims_incorr)
                u_hist_part_sim_prevques_incorr_min[cnt] = min(prev_w2v_pq_sims_incorr)

            if len(user_prev_questions) >= 2:
                prev_idx = 2
                ans_flag = self.user_rec_dict[uid][-1 * prev_idx]
                if ans_flag == 0:
                    ans_flag = -1
                u_hist_ques_sim_prev2ques[cnt] = prev_w2v_qq_sims[prev_idx - 1]
                u_hist_ques_sim_prev2ques_ans[cnt] = prev_w2v_qq_sims[prev_idx - 1] * ans_flag
                prev2_qp_sim = w2v_sim(self.w2v_cname2idx[f'c{cid}'], self.w2v_cname2idx[f'p{self.question2part[user_prev_questions[-2]]}'])
                u_hist_ques_sim_prev2part_ans[cnt] = prev2_qp_sim  # prev_w2v_qp_sims[prev_idx - 1] * ans_flag
                u_hist_part_sim_prev2ques_ans[cnt] = prev_w2v_pq_sims[prev_idx - 1] * ans_flag

            if len(user_prev_questions) >= 3:
                prev_idx = 3
                ans_flag = self.user_rec_dict[uid][-1 * prev_idx]
                if ans_flag == 0:
                    ans_flag = -1
                u_hist_ques_sim_prev3ques[cnt] = prev_w2v_qq_sims[prev_idx - 1]
                u_hist_ques_sim_prev3ques_ans[cnt] = prev_w2v_qq_sims[prev_idx - 1] * ans_flag

            if len(user_prev_questions) >= 4:
                prev_idx = 4
                ans_flag = self.user_rec_dict[uid][-1 * prev_idx]
                if ans_flag == 0:
                    ans_flag = -1
                u_hist_ques_sim_prev4ques[cnt] = prev_w2v_qq_sims[prev_idx - 1]

            if self.lecture_latest_u_dict[uid] != -1:
                u_hist_sim_cid_latest_lec[cnt] = w2v_sim(self.w2v_cname2idx[f'c{cid}'], self.w2v_cname2idx[f'l{self.lecture_latest_u_dict[uid]}'])
                u_hist_sim_cid_latest_lec_part[cnt] = w2v_sim(self.w2v_cname2idx[f'c{cid}'], self.w2v_cname2idx[f'p{self.lecture_id2part[self.lecture_latest_u_dict[uid]]+1}'])

                return

        def add_user_part_record_feat(cnt, part, user_part, user_corr_part):

            u_hist_part_num[cnt] = user_part[part - 1]
            u_hist_part_corr_num[cnt] = user_corr_part[part - 1]

            u_part_1_cnt[cnt] = int(user_part[0])
            u_part_2_cnt[cnt] = int(user_part[1])
            u_part_3_cnt[cnt] = int(user_part[2])
            u_part_4_cnt[cnt] = int(user_part[3])
            u_part_5_cnt[cnt] = int(user_part[4])
            u_part_6_cnt[cnt] = int(user_part[5])
            u_part_7_cnt[cnt] = int(user_part[6])

            u_part_1_corr_rate[cnt] = user_corr_part[0] / float(user_part[0] + 1)
            u_part_2_corr_rate[cnt] = user_corr_part[1] / float(user_part[1] + 1)
            u_part_3_corr_rate[cnt] = user_corr_part[2] / float(user_part[2] + 1)
            u_part_4_corr_rate[cnt] = user_corr_part[3] / float(user_part[3] + 1)
            u_part_5_corr_rate[cnt] = user_corr_part[4] / float(user_part[4] + 1)
            u_part_6_corr_rate[cnt] = user_corr_part[5] / float(user_part[5] + 1)
            u_part_7_corr_rate[cnt] = user_corr_part[6] / float(user_part[6] + 1)
            return

        def add_user_tag_record_feat(cnt, tags, user_tag, user_corr_tag):

            tag_nums = []
            tag_corr_nums = []
            tag_ranks = []
            for t in tags:
                tag_nums.append(user_tag[t])
                tag_corr_nums.append(user_corr_tag[t])
                tag_ranks.append(user_tag.argsort().argsort()[t])
            tag_corr_rates = [tag_corr_nums[i] / tag_nums[i] if tag_nums[i] != 0 else np.nan for i in range(len(tags))]

            u_hist_tag_num[cnt] = sum(tag_nums)
            u_hist_tag_corr_num[cnt] = sum(tag_corr_nums)
            u_hist_tag_corr_num_min[cnt] = min(tag_corr_nums)

            u_hist_tag_rank_min[cnt] = min(tag_ranks)
            u_hist_tag_rank_sum[cnt] = sum(tag_ranks)

            u_hist_tag_corr_rate_min[cnt] = min(tag_corr_rates)
            u_hist_tag_corr_rate_mn[cnt] = sum(tag_corr_rates) / len(tag_corr_rates)
            return

        def add_user_timestamp_feat(cnt, uid, timestamp, user_timestmaps, ets, user_records):
            if len(user_timestmaps) == 0:
                u_prev_difftime[cnt] = -1
                u_prev_lagtime[cnt] = -1

                u_latest5_lagtime_max[cnt] = 0
                u_latest5_et_mn[cnt] = np.nan

                u_latest10_et_max[cnt] = 0
                u_latest10_et_mn[cnt] = np.nan

                u_prev_et_diff_time[cnt] = np.nan
            else:
                difftimes = [0] + [user_timestmaps[i + 1] - user_timestmaps[i] for i in range(len(user_timestmaps) - 1)]
                lagtimes = [-1, -1] + [difftimes[i + 1] - ets[i + 2] for i in range(len(difftimes) - 2)]

                u_prev_difftime[cnt] = timestamp - user_timestmaps[-1]
                if len(user_timestmaps) == 2:
                    u_prev_difftime_2[cnt] = timestamp - user_timestmaps[-2]
                    u_prev_difftime_3[cnt] = -1
                    u_prev_difftime_4[cnt] = -1
                    u_prev_lagtime_2[cnt] = difftimes[-2] - ets[-1]
                    u_prev_lagtime_3[cnt] = -1
                elif len(user_timestmaps) == 3:
                    u_prev_difftime_2[cnt] = timestamp - user_timestmaps[-2]
                    u_prev_difftime_3[cnt] = timestamp - user_timestmaps[-3]
                    u_prev_difftime_4[cnt] = -1
                    u_prev_lagtime_2[cnt] = difftimes[-2] - ets[-1]
                    u_prev_lagtime_3[cnt] = difftimes[-3] - ets[-2]
                elif len(user_timestmaps) >= 4:
                    u_prev_difftime_2[cnt] = timestamp - user_timestmaps[-2]
                    u_prev_difftime_3[cnt] = timestamp - user_timestmaps[-3]
                    u_prev_difftime_4[cnt] = timestamp - user_timestmaps[-4]
                    u_prev_lagtime_2[cnt] = difftimes[-2] - ets[-1]
                    u_prev_lagtime_3[cnt] = difftimes[-3] - ets[-2]

                u_prev_difftime_incorr[cnt] = timestamp - self.user_timestamp_incorr_dict[uid]

                if difftimes[-1] == 0:
                    u_prev_lagtime[cnt] = -1
                else:
                    u_prev_lagtime[cnt] = difftimes[-1] - et

                ets = [i for i in ets if i != -1]  # remove '-1' flag
                if ets == []:
                    ets = [0]

                u_latest5_lagtime_max[cnt] = max(lagtimes[-5:])
                u_latest5_lagtime_median[cnt] = statistics.median(lagtimes[-5:])
                u_latest5_difftime_median[cnt] = statistics.median(difftimes[-5:])

                u_latest10_lagtime_median[cnt] = statistics.median(lagtimes[-10:])
                u_latest10_difftime_median[cnt] = statistics.median(difftimes[-10:])

                u_latest5_et_mn[cnt] = sum(ets[-5:]) / len(ets[-5:])
                u_latest10_et_max[cnt] = max(ets[-10:])
                u_latest10_et_mn[cnt] = sum(ets[-10:]) / len(ets[-10:])

                u_in_seq_lagtime_median[cnt] = statistics.median(lagtimes)
                u_in_seq_difftime_median[cnt] = statistics.median(difftimes)
                u_in_seq_et_max[cnt] = max(ets)
                u_in_seq_et_mn[cnt] = sum(ets) / len(ets)

                u_prev_et_diff_time[cnt] = et / (u_prev_difftime[cnt] + 1)

                difftime_div_in_seq_difftime_median[cnt] = u_prev_difftime[cnt] / (u_in_seq_difftime_median[cnt] + 1)
                difftime_diff_in_seq_lagtime_median[cnt] = u_prev_difftime[cnt] - (u_in_seq_lagtime_median[cnt] + 1)
                difftime_div_in_seq_lagtime_median[cnt] = u_prev_difftime[cnt] / (u_in_seq_lagtime_median[cnt] + 1)

            return

        if add_feat is True and update_dict is False:
            use_columns = self.loop_features
        else:
            use_columns = self.loop_features + self.use_labels

        if add_feat is False and update_dict is True:

            cnt = 0
            for timestamp, ctype, uid, cid, part, tags, hexp, et, tci, ans, uans in df[use_columns].values:

                if ctype == 1:
                    self.lecture_latest_u_dict[uid] = cid
                    self.lecture_type_cnt_u_dict[uid][self.lecture_id2type[cid]] += 1
                    self.lecture_part_cnt_u_dict[uid][self.lecture_id2part[cid]] += 1
                    self.ans_num_from_latest_lecture_u_dict[uid] = 1
                    continue

                user_timestamps = self.user_timestamp_dict[uid]
                if len(user_timestamps) >= 2:
                    difftime = timestamp - user_timestamps[-1]
                    lagtime = (user_timestamps[-1] - user_timestamps[-2]) - et
                elif len(user_timestamps) == 1:
                    difftime = timestamp - user_timestamps[-1]
                    lagtime = -1
                else:
                    difftime = 0
                    lagtime = -1
                self.update_user_log_dict(uid, cid, part, tags, ans, uans, timestamp, et, hexp, difftime, lagtime, tci)
            return

        if add_feat is True:

            rec_num = len(df[df.content_type_id == 0])

            u_cnt = np.zeros(rec_num, dtype=np.int32)
            u_corr_cnt = np.zeros(rec_num, dtype=np.int32)

            u_first_content = np.zeros(rec_num, dtype=np.int32)
            u_first10_corr_rate = np.zeros(rec_num, dtype=np.float32)
            u_first1day_corr_rate = np.zeros(rec_num, dtype=np.float32)
            u_first1week_corr_rate = np.zeros(rec_num, dtype=np.float32)
            u_first1day_cnt = np.zeros(rec_num, dtype=np.int32)
            u_first1week_cnt = np.zeros(rec_num, dtype=np.int32)

            u_uans_rate = np.zeros(rec_num, dtype=np.float32)
            u_incorr_uans_rate = np.zeros(rec_num, dtype=np.float32)
            u_uans_rate_max_diff = np.zeros(rec_num, dtype=np.float32)
            u_incorr_uans_rate_max_diff = np.zeros(rec_num, dtype=np.float32)

            u_cnt_in_session_short = np.zeros(rec_num, dtype=np.uint16)

            u_hade_cnt = np.zeros(rec_num, dtype=np.int32)
            u_hade_corr_cnt = np.zeros(rec_num, dtype=np.int32)

            content_id_diff_abs = np.zeros(rec_num, dtype=np.int32)
            task_container_id_diff_abs = np.zeros(rec_num, dtype=np.int32)

            u_cnt_in_session_short_density = np.zeros(rec_num, dtype=np.float32)
            u_corr_rate_in_session_short = np.zeros(rec_num, dtype=np.float32)

            u_latest10_corr_rate = np.zeros(rec_num, dtype=np.float32)
            u_latest30_corr_rate = np.zeros(rec_num, dtype=np.float32)
            u_latest10_corr_rate_diff_all = np.zeros(rec_num, dtype=np.float32)
            u_latest30_corr_rate_diff_all = np.zeros(rec_num, dtype=np.float32)
            u_in_seq_corr_rate = np.zeros(rec_num, dtype=np.float32)

            seq2dec_w3 = np.zeros(rec_num, dtype=np.float32)
            seq2dec_w7 = np.zeros(rec_num, dtype=np.float32)

            u_hist_queskm100_num = np.zeros(rec_num, dtype=np.uint8)
            u_hist_queskm100_corr_num = np.zeros(rec_num, dtype=np.uint8)

            u_hist_ques_cokm100_num = np.zeros(rec_num, dtype=np.uint8)
            u_hist_ques_cokm100_corr_num = np.zeros(rec_num, dtype=np.uint8)

            u_hist_ques_incokm100_num = np.zeros(rec_num, dtype=np.uint8)
            u_hist_ques_incokm100_corr_num = np.zeros(rec_num, dtype=np.uint8)

            u_hist_ques_in_seq = np.zeros(rec_num, dtype=np.uint8)
            u_hist_ques_in_seq_num = np.zeros(rec_num, dtype=np.uint8)

            u_hist_ques_sim_prevques = np.zeros(rec_num, dtype=np.float32)
            u_hist_ques_sim_prevques_ans = np.zeros(rec_num, dtype=np.float32)
            u_hist_ques_sim_prevpart = np.zeros(rec_num, dtype=np.float32)
            u_hist_ques_sim_prevpart_ans = np.zeros(rec_num, dtype=np.float32)
            u_hist_part_sim_prevques = np.zeros(rec_num, dtype=np.float32)
            u_hist_part_sim_prevques_ans = np.zeros(rec_num, dtype=np.float32)

            u_hist_ques_sim_prevques_mean = np.zeros(rec_num, dtype=np.float32)
            u_hist_ques_sim_prevques_max = np.zeros(rec_num, dtype=np.float32)
            u_hist_ques_sim_prevques_min = np.zeros(rec_num, dtype=np.float32)
            u_hist_part_sim_prevques_mean = np.zeros(rec_num, dtype=np.float32)
            u_hist_part_sim_prevques_max = np.zeros(rec_num, dtype=np.float32)
            u_hist_part_sim_prevques_min = np.zeros(rec_num, dtype=np.float32)

            u_hist_ques_sim_prevques_incorr_mean = np.zeros(rec_num, dtype=np.float32)
            u_hist_ques_sim_prevques_incorr_max = np.zeros(rec_num, dtype=np.float32)
            u_hist_ques_sim_prevques_incorr_min = np.zeros(rec_num, dtype=np.float32)
            u_hist_part_sim_prevques_incorr_mean = np.zeros(rec_num, dtype=np.float32)
            u_hist_part_sim_prevques_incorr_max = np.zeros(rec_num, dtype=np.float32)
            u_hist_part_sim_prevques_incorr_min = np.zeros(rec_num, dtype=np.float32)

            u_hist_ques_sim_prev2ques = np.zeros(rec_num, dtype=np.float32)
            u_hist_ques_sim_prev2ques_ans = np.zeros(rec_num, dtype=np.float32)
            u_hist_ques_sim_prev2part_ans = np.zeros(rec_num, dtype=np.float32)
            u_hist_part_sim_prev2ques_ans = np.zeros(rec_num, dtype=np.float32)

            u_hist_ques_sim_prev3ques = np.zeros(rec_num, dtype=np.float32)
            u_hist_ques_sim_prev3ques_ans = np.zeros(rec_num, dtype=np.float32)

            u_hist_ques_sim_prev4ques = np.zeros(rec_num, dtype=np.float32)

            u_hist_part_num = np.zeros(rec_num, dtype=np.uint16)
            u_hist_part_corr_num = np.zeros(rec_num, dtype=np.uint16)

            u_part_1_cnt = np.zeros(rec_num, dtype=np.uint16)
            u_part_2_cnt = np.zeros(rec_num, dtype=np.uint16)
            u_part_3_cnt = np.zeros(rec_num, dtype=np.uint16)
            u_part_4_cnt = np.zeros(rec_num, dtype=np.uint16)
            u_part_5_cnt = np.zeros(rec_num, dtype=np.uint16)
            u_part_6_cnt = np.zeros(rec_num, dtype=np.uint16)
            u_part_7_cnt = np.zeros(rec_num, dtype=np.uint16)

            u_part_1_corr_rate = np.zeros(rec_num, dtype=np.float32)
            u_part_2_corr_rate = np.zeros(rec_num, dtype=np.float32)
            u_part_3_corr_rate = np.zeros(rec_num, dtype=np.float32)
            u_part_4_corr_rate = np.zeros(rec_num, dtype=np.float32)
            u_part_5_corr_rate = np.zeros(rec_num, dtype=np.float32)
            u_part_6_corr_rate = np.zeros(rec_num, dtype=np.float32)
            u_part_7_corr_rate = np.zeros(rec_num, dtype=np.float32)

            u_hist_tag_num = np.zeros(rec_num, dtype=np.int16)
            u_hist_tag_corr_num = np.zeros(rec_num, dtype=np.int16)
            u_hist_tag_corr_num_min = np.zeros(rec_num, dtype=np.int16)
            u_hist_tag_rank_min = np.zeros(rec_num, dtype=np.int16)
            u_hist_tag_rank_sum = np.zeros(rec_num, dtype=np.uint16)
            u_hist_tag_corr_rate_min = np.zeros(rec_num, dtype=np.float32)
            u_hist_tag_corr_rate_mn = np.zeros(rec_num, dtype=np.float32)

            u_hist_lec_part_cnt = np.zeros(rec_num, dtype=np.int16)
            u_hist_lec_type_1_cnt = np.zeros(rec_num, dtype=np.int16)
            u_hist_lec_type_2_cnt = np.zeros(rec_num, dtype=np.int16)
            u_hist_ans_num_from_lec = np.zeros(rec_num, dtype=np.int16)
            u_hist_sim_cid_latest_lec = np.zeros(rec_num, dtype=np.float32)
            u_hist_sim_cid_latest_lec_part = np.zeros(rec_num, dtype=np.float32)

            u_prev_difftime = np.zeros(rec_num, dtype=np.int64)
            u_prev_difftime_2 = np.zeros(rec_num, dtype=np.int64)
            u_prev_difftime_3 = np.zeros(rec_num, dtype=np.int64)
            u_prev_difftime_4 = np.zeros(rec_num, dtype=np.int64)
            u_prev_difftime_incorr = np.zeros(rec_num, dtype=np.int64)
            u_prev_lagtime = np.zeros(rec_num, dtype=np.int64)
            u_prev_lagtime_2 = np.zeros(rec_num, dtype=np.int64)
            u_prev_lagtime_3 = np.zeros(rec_num, dtype=np.int64)
            u_hist_et_sum_div_cnt = np.zeros(rec_num, dtype=np.float32)
            u_hist_et_sum_div_corr_cnt = np.zeros(rec_num, dtype=np.float32)

            u_latest5_lagtime_max = np.zeros(rec_num, dtype=np.float32)
            u_latest5_lagtime_median = np.zeros(rec_num, dtype=np.float32)
            u_latest5_difftime_median = np.zeros(rec_num, dtype=np.float32)

            u_latest10_lagtime_median = np.zeros(rec_num, dtype=np.float32)
            u_latest10_difftime_median = np.zeros(rec_num, dtype=np.float32)

            u_latest5_et_mn = np.zeros(rec_num, dtype=np.float32)
            u_latest10_et_mn = np.zeros(rec_num, dtype=np.float32)
            u_latest10_et_max = np.zeros(rec_num, dtype=np.uint32)

            u_in_seq_lagtime_median = np.zeros(rec_num, dtype=np.uint32)
            u_in_seq_difftime_median = np.zeros(rec_num, dtype=np.uint32)
            u_in_seq_et_mn = np.zeros(rec_num, dtype=np.float32)
            u_in_seq_et_max = np.zeros(rec_num, dtype=np.uint32)

            u_prev_et_diff_time = np.zeros(rec_num, dtype=np.float32)

            difftime_div_in_seq_difftime_median = np.zeros(rec_num, dtype=np.float32)
            difftime_diff_in_seq_lagtime_median = np.zeros(rec_num, dtype=np.int64)
            difftime_div_in_seq_lagtime_median = np.zeros(rec_num, dtype=np.float32)

            cnt = 0
            for row in tqdm(df[use_columns].values):

                if update_dict is True:
                    timestamp, ctype, uid, cid, part, tags, hexp, et, tci, ans, uans = row
                else:
                    timestamp, ctype, uid, cid, part, tags, hexp, et, tci = row

                if ctype == 1:
                    self.lecture_latest_u_dict[uid] = cid
                    self.lecture_type_cnt_u_dict[uid][self.lecture_id2type[cid]] += 1
                    self.lecture_part_cnt_u_dict[uid][self.lecture_id2part[cid]] += 1
                    self.ans_num_from_latest_lecture_u_dict[uid] = 1
                    continue

                # add feat
                u_corr_cnt[cnt] = self.user_corr_cnt_dict[uid]
                u_cnt[cnt] = self.user_cnt_dict[uid]

                u_first_content[cnt] = self.user_first_content_dict[uid]
                u_first10_corr_rate[cnt] = self.user_first10_corr_rate_dict[uid]
                u_first1day_corr_rate[cnt] = self.user_first1day_corr_rate_dict[uid]
                u_first1week_corr_rate[cnt] = self.user_first1week_corr_rate_dict[uid]
                u_first1day_cnt[cnt] = self.user_first1day_cnt_dict[uid]
                u_first1week_cnt[cnt] = self.user_first1week_cnt_dict[uid]

                incorr_cnt = self.user_cnt_dict[uid] - self.user_corr_cnt_dict[uid]
                cid_uans_rate = self.user_answer_u_dict[uid][self.content_id2corrans[cid]] / (self.user_cnt_dict[uid] + 1)
                cid_incorr_uans_rate = self.incorr_user_answer_u_dict[uid][self.content_id2corrans[cid]] / (incorr_cnt + 1)

                uans_max_rate = max(self.user_answer_u_dict[uid]) / (self.user_cnt_dict[uid] + 1)
                incorr_uans_max_rate = max(self.incorr_user_answer_u_dict[uid]) / (incorr_cnt + 1)

                u_uans_rate[cnt] = cid_uans_rate
                u_incorr_uans_rate[cnt] = cid_incorr_uans_rate
                u_uans_rate_max_diff[cnt] = uans_max_rate - cid_uans_rate
                u_incorr_uans_rate_max_diff[cnt] = incorr_uans_max_rate - cid_incorr_uans_rate

                user_records = self.user_rec_dict[uid]
                add_user_latest_record_feat(cnt, user_records)

                add_user_ques_record_feat(cnt, uid, cid, part, tags, val=val)

                u_hade_corr_cnt[cnt] = self.hadexp_sum_u_dict[uid]
                u_hade_cnt[cnt] = self.hadexp_cnt_u_dict[uid]

                if len(self.user_prev_ques_dict[uid]) > 0:
                    content_id_diff_abs[cnt] = abs(cid - self.user_prev_ques_dict[uid][-1])
                task_container_id_diff_abs[cnt] = abs(tci - self.user_latest_tci_dict[uid])

                user_part = self.part_u_dict[uid]
                user_corr_part = self.part_corr_u_dict[uid]
                add_user_part_record_feat(cnt, part, user_part, user_corr_part)

                user_tag = self.tag_u_dict[uid]
                user_corr_tag = self.tag_corr_u_dict[uid]
                add_user_tag_record_feat(cnt, tags, user_tag, user_corr_tag)

                u_hist_lec_part_cnt[cnt] = self.lecture_part_cnt_u_dict[uid][part - 1]
                u_hist_lec_type_1_cnt[cnt] = self.lecture_type_cnt_u_dict[uid][1]
                u_hist_lec_type_2_cnt[cnt] = self.lecture_type_cnt_u_dict[uid][2]

                u_hist_ans_num_from_lec[cnt] = self.ans_num_from_latest_lecture_u_dict[uid]

                user_timestmaps = self.user_timestamp_dict[uid]
                ets = self.user_et_dict[uid]

                add_user_timestamp_feat(cnt, uid, timestamp, user_timestmaps, ets, user_records)
                u_hist_et_sum_div_cnt[cnt] = self.user_et_sum_dict[uid] / (self.user_cnt_dict[uid] + 1)
                u_hist_et_sum_div_corr_cnt[cnt] = self.user_et_sum_dict[uid] / (self.user_corr_cnt_dict[uid] + 1)

                if u_prev_difftime[cnt] > self.session_short_th:
                    self.user_prev_session_short_ts_dict[uid] = timestamp
                    self.user_corr_cnt_in_session_short_dict[uid] = 0
                    self.user_cnt_in_session_short_dict[uid] = 0
                u_ts_in_session_short = timestamp - self.user_prev_session_short_ts_dict[uid]

                u_cnt_in_session_short[cnt] = self.user_cnt_in_session_short_dict[uid]
                if u_ts_in_session_short > 0:
                    u_cnt_in_session_short_density[cnt] = u_cnt_in_session_short[cnt] / u_ts_in_session_short

                if u_cnt_in_session_short[cnt] > 0:
                    u_corr_rate_in_session_short[cnt] = self.user_corr_cnt_in_session_short_dict[uid] / u_cnt_in_session_short[cnt]
                else:
                    u_corr_rate_in_session_short[cnt] = np.nan

                # update dict
                if update_dict is True:
                    self.update_user_log_dict(
                        uid, cid, part, tags, ans, uans, timestamp, et, hexp,
                        u_prev_difftime[cnt], u_prev_lagtime[cnt], tci
                    )

                cnt += 1

            user_feats_df = pd.DataFrame({
                #
                # user_history
                #
                'u_cnt': u_cnt,
                'u_corr_cnt': u_corr_cnt,
                'u_cnt_in_session_short': u_cnt_in_session_short,
                'u_cnt_in_session_short_density': u_cnt_in_session_short_density,
                'u_latest10_corr_rate': u_latest10_corr_rate,
                'u_latest30_corr_rate': u_latest30_corr_rate,
                'u_latest10_corr_rate_diff_all': u_latest10_corr_rate_diff_all,
                'u_latest30_corr_rate_diff_all': u_latest30_corr_rate_diff_all,
                'u_in_seq_corr_rate': u_in_seq_corr_rate,
                'u_corr_rate_in_session_short': u_corr_rate_in_session_short,
                'seq2dec_w3': seq2dec_w3,
                'seq2dec_w7': seq2dec_w7,
                'content_id_diff_abs': content_id_diff_abs,
                'task_container_id_diff_abs': task_container_id_diff_abs,
                'u_first_content': u_first_content,
                'u_first10_corr_rate': u_first10_corr_rate,
                'u_first1day_corr_rate': u_first1day_corr_rate,
                'u_first1week_corr_rate': u_first1week_corr_rate,
                'u_first1day_cnt': u_first1day_cnt,
                'u_first1week_cnt': u_first1week_cnt,
                'u_uans_rate': u_uans_rate,
                'u_incorr_uans_rate': u_incorr_uans_rate,
                'u_uans_rate_max_diff': u_uans_rate_max_diff,
                'u_incorr_uans_rate_max_diff': u_incorr_uans_rate_max_diff,
                #
                # user_history_hade
                #
                'u_hade_cnt': u_hade_cnt,
                'u_hade_corr_cnt': u_hade_corr_cnt,
                #
                # user_history_question
                #
                'u_hist_ques_in_seq': u_hist_ques_in_seq,
                'u_hist_ques_in_seq_num': u_hist_ques_in_seq_num,
                'u_hist_queskm100_num': u_hist_queskm100_num,
                'u_hist_queskm100_corr_num': u_hist_queskm100_corr_num,
                'u_hist_ques_cokm100_num': u_hist_ques_cokm100_num,
                'u_hist_ques_cokm100_corr_num': u_hist_ques_cokm100_corr_num,
                'u_hist_ques_incokm100_num': u_hist_ques_incokm100_num,
                'u_hist_ques_incokm100_corr_num': u_hist_ques_incokm100_corr_num,
                'u_hist_ques_sim_prevques': u_hist_ques_sim_prevques,
                'u_hist_ques_sim_prevques_ans': u_hist_ques_sim_prevques_ans,
                'u_hist_ques_sim_prevpart': u_hist_ques_sim_prevpart,
                'u_hist_ques_sim_prevpart_ans': u_hist_ques_sim_prevpart_ans,
                'u_hist_part_sim_prevques': u_hist_part_sim_prevques,
                'u_hist_part_sim_prevques_ans': u_hist_part_sim_prevques_ans,
                #
                'u_hist_ques_sim_prevques_mean': u_hist_ques_sim_prevques_mean,
                'u_hist_ques_sim_prevques_max': u_hist_ques_sim_prevques_max,
                'u_hist_ques_sim_prevques_min': u_hist_ques_sim_prevques_min,
                'u_hist_part_sim_prevques_mean': u_hist_part_sim_prevques_mean,
                'u_hist_part_sim_prevques_max': u_hist_part_sim_prevques_max,
                'u_hist_part_sim_prevques_min': u_hist_part_sim_prevques_min,
                #
                'u_hist_ques_sim_prevques_incorr_mean': u_hist_ques_sim_prevques_incorr_mean,
                'u_hist_ques_sim_prevques_incorr_max': u_hist_ques_sim_prevques_incorr_max,
                'u_hist_ques_sim_prevques_incorr_min': u_hist_ques_sim_prevques_incorr_min,
                'u_hist_part_sim_prevques_incorr_mean': u_hist_part_sim_prevques_incorr_mean,
                'u_hist_part_sim_prevques_incorr_max': u_hist_part_sim_prevques_incorr_max,
                'u_hist_part_sim_prevques_incorr_min': u_hist_part_sim_prevques_incorr_min,
                #
                'u_hist_ques_sim_prev2ques': u_hist_ques_sim_prev2ques,
                'u_hist_ques_sim_prev2ques_ans': u_hist_ques_sim_prev2ques_ans,
                'u_hist_ques_sim_prev2part_ans': u_hist_ques_sim_prev2part_ans,
                'u_hist_part_sim_prev2ques_ans': u_hist_part_sim_prev2ques_ans,
                #
                'u_hist_ques_sim_prev3ques': u_hist_ques_sim_prev3ques,
                'u_hist_ques_sim_prev3ques_ans': u_hist_ques_sim_prev3ques_ans,
                #
                'u_hist_ques_sim_prev4ques': u_hist_ques_sim_prev4ques,
                #
                #
                # user_history_part
                #
                'u_hist_part_num': u_hist_part_num,
                'u_hist_part_corr_num': u_hist_part_corr_num,
                'u_part_1_cnt': u_part_1_cnt,
                'u_part_2_cnt': u_part_2_cnt,
                'u_part_3_cnt': u_part_3_cnt,
                'u_part_4_cnt': u_part_4_cnt,
                'u_part_5_cnt': u_part_5_cnt,
                'u_part_6_cnt': u_part_6_cnt,
                'u_part_7_cnt': u_part_7_cnt,
                'u_part_1_corr_rate': u_part_1_corr_rate,
                'u_part_2_corr_rate': u_part_2_corr_rate,
                'u_part_3_corr_rate': u_part_3_corr_rate,
                'u_part_4_corr_rate': u_part_4_corr_rate,
                'u_part_5_corr_rate': u_part_5_corr_rate,
                'u_part_6_corr_rate': u_part_6_corr_rate,
                'u_part_7_corr_rate': u_part_7_corr_rate,
                #
                # user_history_tag
                #
                'u_hist_tag_num': u_hist_tag_num,
                'u_hist_tag_corr_num': u_hist_tag_corr_num,
                'u_hist_tag_corr_num_min': u_hist_tag_corr_num_min,
                'u_hist_tag_rank_min': u_hist_tag_rank_min,
                'u_hist_tag_rank_sum': u_hist_tag_rank_sum,
                'u_hist_tag_corr_rate_min': u_hist_tag_corr_rate_min,
                'u_hist_tag_corr_rate_mn': u_hist_tag_corr_rate_mn,
                #
                # user_history_lecture
                #
                'u_hist_lec_part_cnt': u_hist_lec_part_cnt,
                'u_hist_lec_type_1_cnt': u_hist_lec_type_1_cnt,
                'u_hist_lec_type_2_cnt': u_hist_lec_type_2_cnt,
                'u_hist_ans_num_from_lec': u_hist_ans_num_from_lec,
                'u_hist_sim_cid_latest_lec': u_hist_sim_cid_latest_lec,
                'u_hist_sim_cid_latest_lec_part': u_hist_sim_cid_latest_lec_part,
                #
                # user_history_time
                #
                'u_prev_difftime': u_prev_difftime,
                'u_prev_difftime_2': u_prev_difftime_2,
                'u_prev_difftime_3': u_prev_difftime_3,
                'u_prev_difftime_4': u_prev_difftime_4,
                'u_prev_difftime_incorr': u_prev_difftime_incorr,
                'u_prev_lagtime': u_prev_lagtime,
                'u_prev_lagtime_2': u_prev_lagtime_2,
                'u_prev_lagtime_3': u_prev_lagtime_3,
                'u_hist_et_sum_div_cnt': u_hist_et_sum_div_cnt,
                'u_hist_et_sum_div_corr_cnt': u_hist_et_sum_div_corr_cnt,
                'u_latest5_lagtime_max': u_latest5_lagtime_max,
                'u_latest5_lagtime_median': u_latest5_lagtime_median,
                'u_latest5_difftime_median': u_latest5_difftime_median,
                'u_latest10_lagtime_median': u_latest10_lagtime_median,
                'u_latest10_difftime_median': u_latest10_difftime_median,
                'u_latest5_et_mn': u_latest5_et_mn,
                'u_latest10_et_mn': u_latest10_et_mn,
                'u_latest10_et_max': u_latest10_et_max,
                'u_in_seq_et_mn': u_in_seq_et_mn,
                'u_in_seq_et_max': u_in_seq_et_max,
                'u_prev_et_diff_time': u_prev_et_diff_time,
                'difftime_div_in_seq_difftime_median': difftime_div_in_seq_difftime_median,
                'difftime_diff_in_seq_lagtime_median': difftime_diff_in_seq_lagtime_median,
                'difftime_div_in_seq_lagtime_median': difftime_div_in_seq_lagtime_median,
            })
            user_feats_df = self.dataframe_process(user_feats_df)
            return user_feats_df

    def reduce_svd(self, features, n_components):
        dr_model = TruncatedSVD(n_components=n_components, random_state=46)
        features_dr = dr_model.fit_transform(features)
        return features_dr

    def fe_agg(self, X_tra_wo_lec, cols, target, table_name=''):

        colname = 'xxx'.join(cols)
        agg = X_tra_wo_lec[cols + [target]].groupby(cols).agg(['count', 'std']).reset_index()
        agg.columns = cols + [f'{colname}__count{table_name}', f'{colname}__std{table_name}']

        agg.to_feather(f'../save/features_{FOLD_NAME}/{colname}__{target}_agg_count_std{table_name}.feather')
        return agg

    def fe_unique_user(self, X_tra_wo_lec, cols, table_name=''):

        colname = 'xxx'.join(cols)
        agg = X_tra_wo_lec[cols + ['user_id']].groupby(cols)['user_id'].nunique().reset_index()
        agg.columns = cols + [f'{colname}__unique_user{table_name}']

        agg.to_feather(f'../save/features_{FOLD_NAME}/{colname}__unique_user{table_name}.feather')
        return agg

    def fe_te_sm(self, df, cols, target, mn, table_name=''):

        colname = 'xxx'.join(cols)
        fname = f'{colname}__{target}_sm{self.smooth}{table_name}'

        agg = df[cols + [target]].groupby(cols).agg(['mean', 'count']).reset_index()
        agg.columns = cols + [f'{colname}__{target}', f'{colname}__count']
        agg[fname] = (agg[f'{colname}__count'] * agg[f'{colname}__{target}'] + self.smooth * mn) / (agg[f'{colname}__count'] + self.smooth)
        agg = agg[cols + [fname]]

        agg.to_feather(f'../save/features_{FOLD_NAME}/{colname}__{target}_sm{self.smooth}{table_name}.feather')
        return agg

    def fe_ent(self, X_tra_wo_lec, cols, target):

        colname = 'xxx'.join(cols)
        agg = X_tra_wo_lec[cols + [target]].groupby(cols)[target].agg(
            lambda x: multinomial.entropy(1, x.value_counts(normalize=True))
        ).reset_index()
        agg.columns = cols + [f'{colname}__entropy__{target}']

        agg.to_feather(f'../save/features_{FOLD_NAME}/{colname}__entropy__{target}.feather')
        return agg

    def fe_svd(self, X_tra_wo_lec, cols):

        colname = 'xxx'.join(cols)
        svd_dim = 5
        ids = []
        sequences = []
        for c, row in tqdm(X_tra_wo_lec[['user_id'] + cols].groupby(cols)):
            ids.append(c)
            sequences.append(row['user_id'].values.tolist())
        mlb = MultiLabelBinarizer()
        tags_mlb = mlb.fit_transform(sequences)
        svd = self.reduce_svd(tags_mlb, n_components=svd_dim)
        svd = pd.DataFrame(svd).add_prefix(f'{colname}__svd_')
        if len(cols) == 1:
            svd[cols[0]] = ids
        else:
            svd[cols] = ids

        svd.to_feather(f'../save/features_{FOLD_NAME}/{colname}__svd_feat.feather')
        return svd

    def fe_tag_te_agg(self, X_tra_wo_lec, question):

        tmp = {i: x.split() for i, x in enumerate(question['tags'].fillna('999').values)}
        tmp = {'categories': tmp}
        data2 = pd.DataFrame.from_dict(tmp)
        data3 = data2['categories'].apply(Counter)

        tag_df = pd.DataFrame.from_records(data3).fillna(value=0).astype('int8').add_prefix('tag_').reset_index().rename(columns={'index': 'question_id'})
        tag_list = tag_df.columns.values[1:].tolist()
        rows = []
        for tid in tqdm(tag_list):
            tmp = pd.merge(X_tra_wo_lec[['content_id', 'answered_correctly']], tag_df[['question_id', tid]], left_on='content_id', right_on='question_id', how='left')
            mn, cnt = tmp[tmp[tid] == 1].answered_correctly.mean(), tmp[tmp[tid] == 1].answered_correctly.count()
            rows.append([tid, mn, cnt])
        tag_stats_df = pd.DataFrame(rows, columns=['tid', 'mn', 'cnt'])
        tag_mn = {i.split('_')[-1]: j for i, j in tag_stats_df[['tid', 'mn']].values}
        tag_scores = [np.array([tag_mn[j] for j in i.split()]) for i in question['tags'].fillna('999').values]

        question['tags_max'] = [i.max() for i in tag_scores]
        question['tags_min'] = [i.min() for i in tag_scores]
        question['tags_cnt'] = [len(i) for i in tag_scores]
        question['tags_mean'] = [i.mean() for i in tag_scores]
        question['tags_std'] = [i.std() for i in tag_scores]

        content_features = []
        content_features += ['tags_max', 'tags_min', 'tags_cnt', 'tags_mean', 'tags_std']
        question = question[['question_id'] + content_features]
        question.columns = ['content_id'] + content_features

        question.to_feather(f'../save/features_{FOLD_NAME}/content_id__tag_feat.feather')
        return question

    def fe_content_session_border_feature(self, X_tra_wo_lec):

        rows = []
        for uid, user_df in tqdm(X_tra_wo_lec[['user_id', 'content_id', 'timestamp']].groupby('user_id')):
            user_df = user_df.reset_index(drop=True)
            user_df['user_timestamp_diff'] = user_df['timestamp'].diff()
            user_df['user_timestamp_diff'] = user_df['user_timestamp_diff'].fillna('0.0').astype('float32')
            user_df['user_session_start'] = (user_df['user_timestamp_diff'] > self.session_short_th) * 1
            user_df['user_session_end'] = user_df['user_session_start'].shift(-1).fillna(0.0)
            user_df.loc[0, 'user_session_start'] = 1.0
            rows.extend(user_df.values)
        session_df = pd.DataFrame(rows, columns=['user_id', 'content_id', 'timestamp', 'user_timestamp_diff', 'user_session_start', 'user_session_end'])
        session_df = session_df.astype({
            'user_id': 'int',
            'content_id': 'int',
            'timestamp': 'int',
        })

        smooth = 10
        tar = 'user_session_start'
        mn = session_df[tar].mean()
        agg = session_df.groupby('content_id')[tar].agg(['count', 'mean']).reset_index()
        agg['mean_smooth'] = (agg['count'] * agg['mean'] + smooth * mn) / (agg['count'] + smooth)

        feat = pd.concat([
            agg[['content_id']], agg[['mean_smooth']].add_prefix(f'{tar}__')
        ], axis=1)

        tar = 'user_session_end'
        mn = session_df[tar].mean()
        agg = session_df.groupby('content_id')[tar].agg(['count', 'mean']).reset_index()
        agg['mean_smooth'] = (agg['count'] * agg['mean'] + smooth * mn) / (agg['count'] + smooth)

        feat = pd.concat([
            feat, agg[['mean_smooth']].add_prefix(f'{tar}__')
        ], axis=1)

        feat.to_feather(f'../save/features_{FOLD_NAME}/content_session_border_feat.feather')
        return feat

    def fe_content_order_feature(self, X_tra_wo_lec):
        rows = []
        for uid, user_df in tqdm(X_tra_wo_lec[['user_id', 'content_id']].groupby('user_id')):
            user_df['order'] = user_df.reset_index(drop=True).index + 1
            rows.extend(user_df.values)
        order_df = pd.DataFrame(rows, columns=['user_id', 'content_id', 'order'])

        agg_funcs = ['mean', 'median', 'max', 'min', 'std']
        agg = order_df.groupby('content_id')['order'].agg(agg_funcs).reset_index()
        feat = pd.concat([
            agg[['content_id']], agg[agg_funcs].add_prefix('content_order_')
        ], axis=1)

        feat.to_feather(f'../save/features_{FOLD_NAME}/content_order_feat.feather')
        return feat

    def extract_content_id_feat(
        self, X_tra_wo_lec, repeat, et_table, question,
        w2v_features, g_features, corr_g_features, ge_dw_features
    ):

        col = 'content_id'
        tar = 'answered_correctly'
        wtar = 'weighted_answered_correctly'
        ws = 'weighted_score'
        ua = 'user_answer'
        qet = 'question_elapsed_time'

        features = pd.DataFrame(X_tra_wo_lec[col].unique(), columns=[col]).sort_values(col).reset_index(drop=True)

        if f'{col}__count' in self.use_features or f'{col}__std' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_agg_count_std{table_name}.feather'
            if os.path.exists(fpath):
                agg_feat = pd.read_feather(fpath)
            else:
                agg_feat = self.fe_agg(X_tra_wo_lec, [col], target=tar)
            features = pd.merge(features, agg_feat, on=col, how='left')

        if f'{col}__count_repeat' in self.use_features or f'{col}__std_repeat' in self.use_features:
            table_name = '_repeat'
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_agg_count_std{table_name}.feather'
            if os.path.exists(fpath):
                agg_feat = pd.read_feather(fpath)
            else:
                agg_feat = self.fe_agg(repeat, [col], target=tar, table_name=table_name)
            features = pd.merge(features, agg_feat, on=col, how='left')

        if f'{col}__unique_user' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__unique_user.feather'
            if os.path.exists(fpath):
                agg_feat = pd.read_feather(fpath)
            else:
                agg_feat = self.fe_unique_user(X_tra_wo_lec, [col])
            features = pd.merge(features, agg_feat, on=col, how='left')

        if f'{col}__answered_correctly_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(X_tra_wo_lec, [col], target=tar, mn=self.train_mn)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__weighted_answered_correctly_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{wtar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(X_tra_wo_lec, [col], target=wtar, mn=self.train_wans_mn)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__weighted_score_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{ws}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(X_tra_wo_lec, [col], target=ws, mn=self.train_ws_mn)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__answered_correctly_sm5_repeat' in self.use_features:
            table_name = '_repeat'
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(repeat, [col], target=tar, mn=self.train_mn, table_name=table_name)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__weighted_answered_correctly_sm5_repeat' in self.use_features:
            table_name = '_repeat'
            fpath = f'../save/features_{FOLD_NAME}/{col}__{wtar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(repeat, [col], target=wtar, mn=self.train_wans_mn, table_name=table_name)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__weighted_score_sm5_repeat' in self.use_features:
            table_name = '_repeat'
            fpath = f'../save/features_{FOLD_NAME}/{col}__{ws}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(repeat, [col], target=ws, mn=self.train_ws_mn, table_name=table_name)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__entropy__user_answer' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__entropy__{ua}.feather'
            if os.path.exists(fpath):
                ent_feat = pd.read_feather(fpath)
            else:
                ent_feat = self.fe_ent(X_tra_wo_lec, [col], target=ua)
            features = pd.merge(features, ent_feat, on=col, how='left')

        if f'{col}__entropy__answered_correctly' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__entropy__{tar}.feather'
            if os.path.exists(fpath):
                ent_feat = pd.read_feather(fpath)
            else:
                ent_feat = self.fe_ent(X_tra_wo_lec, [col], target=tar)
            features = pd.merge(features, ent_feat, on=col, how='left')

        if f'{col}__question_elapsed_time_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{qet}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(et_table, [col], target=qet, mn=et_table[qet].mean())
            features = pd.merge(features, te_feat, on=col, how='left')

        if 'tags_max' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/{col}__tag_feat.feather'
            if os.path.exists(fpath):
                tag_feat = pd.read_feather(fpath)
            else:
                tag_feat = self.fe_tag_te_agg(X_tra_wo_lec, question)
            features = pd.merge(features, tag_feat, on=col, how='left')

        if len(set([f'{col}__svd_{i}' for i in range(5)]) & set(self.use_features)) > 0:
            fpath = f'../save/features_{FOLD_NAME}/{col}__svd_feat.feather'
            if os.path.exists(fpath):
                svd_feat = pd.read_feather(fpath)
            else:
                svd_feat = self.fe_svd(X_tra_wo_lec, [col])
            features = pd.merge(features, svd_feat, on=col, how='left')

        if len(set([f'w2v_{i}' for i in range(5)]) & set(self.use_features)) > 0:
            features = pd.merge(features, w2v_features, on=col, how='left')

        if len(set([f'gsvd_{i}' for i in range(6)]) & set(self.use_features)) > 0:
            features = pd.merge(features, g_features, on=col, how='left')

        if len(set([f'corr_gsvd_{i}' for i in range(6)]) & set(self.use_features)) > 0:
            features = pd.merge(features, corr_g_features, on=col, how='left')

        if len(set([f'ge_dw_svd_{i}' for i in range(5)]) & set(self.use_features)) > 0:
            features = pd.merge(features, ge_dw_features, on=col, how='left')

        if len(set([f'ge_s2v_svd_{i}' for i in range(5)]) & set(self.use_features)) > 0:
            features = pd.merge(features, ge_s2v_features, on=col, how='left')

        if len(set([f'content_id_tsne_{i}' for i in range(2)]) & set(self.use_features)) > 0:
            ctsne_features = pd.read_feather(content_tsne_path)
            features = pd.merge(features, ctsne_features, on=col, how='left')

        if 'user_session_start__mean_smooth' in self.use_features:
            fpath = f'../save/features_{FOLD_NAME}/content_session_border_feat.feather'
            if os.path.exists(fpath):
                feat = pd.read_feather(fpath)
            else:
                feat = self.fe_content_session_border_feature(X_tra_wo_lec)
            features = pd.merge(features, feat, on=col, how='left')

        if len(set(['content_order_mean', 'content_order_median', 'content_order_max', 'content_order_min', 'content_order_std']) & set(self.use_features)) > 0:
            fpath = f'../save/features_{FOLD_NAME}/content_order_feat.feather'
            if os.path.exists(fpath):
                feat = pd.read_feather(fpath)
            else:
                feat = self.fe_content_order_feature(X_tra_wo_lec)
            features = pd.merge(features, feat, on=col, how='left')

        if len(set(content_graph_metric_features) & set(self.use_features)) > 0:
            features = pd.merge(features, graph_metrics_features, on=col, how='left')

        # part features
        part_feat = self.extract_part_feat(X_tra_wo_lec, repeat, et_table)
        features['part'] = features['content_id'].map(self.question2part)
        features = pd.merge(features, part_feat, on=['part'], how='left')

        feature_list = [col] + [i for i in features.columns.tolist() if i in self.use_features]
        feature_list = list(set(feature_list))
        features = features[feature_list]
        cols = ['content_id'] + [i for i in features.columns.sort_values().tolist() if i != 'content_id']
        self.content_id_df = features[cols]
        self.content_id_df = self.reduce_mem_usage(self.content_id_df)
        return

    def extract_part_feat(self, X_tra_wo_lec, repeat, et_table):

        col = 'part'
        tar = 'answered_correctly'
        wtar = 'weighted_answered_correctly'
        ws = 'weighted_score'
        qet = 'question_elapsed_time'

        features = pd.DataFrame(X_tra_wo_lec[col].unique(), columns=[col]).sort_values(col).reset_index(drop=True)

        if f'{col}__count' in self.use_features or f'{col}__std' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_agg_count_std{table_name}.feather'
            if os.path.exists(fpath):
                agg_feat = pd.read_feather(fpath)
            else:
                agg_feat = self.fe_agg(X_tra_wo_lec, [col], target=tar)
            features = pd.merge(features, agg_feat, on=col, how='left')

        # te
        if f'{col}__answered_correctly_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(X_tra_wo_lec, [col], target=tar, mn=self.train_mn)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__weighted_answered_correctly_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{wtar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(X_tra_wo_lec, [col], target=wtar, mn=self.train_wans_mn)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__weighted_score_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{ws}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(X_tra_wo_lec, [col], target=ws, mn=self.train_ws_mn)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__answered_correctly_sm5_repeat' in self.use_features:
            table_name = '_repeat'
            fpath = f'../save/features_{FOLD_NAME}/{col}__{tar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(repeat, [col], target=tar, mn=self.train_mn, table_name=table_name)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__weighted_answered_correctly_sm5_repeat' in self.use_features:
            table_name = '_repeat'
            fpath = f'../save/features_{FOLD_NAME}/{col}__{wtar}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(repeat, [col], target=wtar, mn=self.train_wans_mn, table_name=table_name)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__weighted_score_sm5_repeat' in self.use_features:
            table_name = '_repeat'
            fpath = f'../save/features_{FOLD_NAME}/{col}__{ws}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(repeat, [col], target=ws, mn=self.train_ws_mn, table_name=table_name)
            features = pd.merge(features, te_feat, on=col, how='left')

        if f'{col}__question_elapsed_time_sm5' in self.use_features:
            table_name = ''
            fpath = f'../save/features_{FOLD_NAME}/{col}__{qet}_sm{self.smooth}{table_name}.feather'
            if os.path.exists(fpath):
                te_feat = pd.read_feather(fpath)
            else:
                te_feat = self.fe_te_sm(et_table, [col], target=qet, mn=et_table[qet].mean())
            features = pd.merge(features, te_feat, on=col, how='left')

        if len(set([f'{col}__svd_{i}' for i in range(5)]) & set(self.use_features)) > 0:
            fpath = f'../save/features_{FOLD_NAME}/{col}__svd_feat.feather'
            if os.path.exists(fpath):
                svd_feat = pd.read_feather(fpath)
            else:
                svd_feat = self.fe_svd(X_tra_wo_lec, [col])
            features = pd.merge(features, svd_feat, on=col, how='left')

        return features

    def reduce_mem_usage(self, df):
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


if __name__ == "__main__":

    mlflow.set_experiment(EXP_NAME)
    mlflow.start_run()
    run_id = mlflow.active_run().info.run_id

    if not os.path.exists(f'../save/{EXP_NAME}_{run_id}/'):
        os.mkdir(f'../save/{EXP_NAME}_{run_id}/')
        os.mkdir(f'../save/{EXP_NAME}_{run_id}/features')
        shutil.copy(f'{EXP_DIR}/{EXP_CONFIG}/config.py', f'../save/{EXP_NAME}_{run_id}/config.py')
        shutil.copy('train_catboost.py', f'../save/{EXP_NAME}_{run_id}/train_catboost.py')
    logging.basicConfig(filename=f'../save/{EXP_NAME}_{run_id}/logger.log', level=logging.INFO)

    t = Timer()
    with t.timer(f'fix seed RANDOM_STATE:{RANDOM_STATE}'):
        seed_everything(RANDOM_STATE)

    with t.timer('read data'):
        folds = pd.read_feather(f'{FOLD_DIR}/train_folds_{FOLD_NAME}_v2_{RANDOM_STATE}.feather')
        train = pd.read_feather(f'{INPUT_DIR}/train_v2.feather')
        question = pd.read_csv(f'{INPUT_DIR}/questions.csv')
        lecture = pd.read_csv(f'{INPUT_DIR}/lectures.csv')
        le = preprocessing.LabelEncoder()
        lecture['type_of'] = le.fit_transform(lecture['type_of'])

    with t.timer('Data split'):
        X_tra = train[train.row_id.isin(folds[folds.val == 0].row_id)]
        X_val = train[train.row_id.isin(folds[folds.val == 1].row_id)]
        X_tra_wo_lec = X_tra[X_tra.content_type_id == 0]
        X_val_wo_lec = X_val[X_val.content_type_id == 0]
        logging.info(f'X_tra:{len(X_tra)}, X_val:{len(X_val)}, X_tra_wo_lec{len(X_tra_wo_lec),}, X_val_wo_lec:{len(X_val_wo_lec)}')

    del train
    del folds
    gc.collect()

    with t.timer('Preprocess'):

        # preprocess
        question['lr'] = question['part'].apply(part2lr)

        X_tra = merge_questions_w_prep(X_tra, question)
        X_val = merge_questions_w_prep(X_val, question)

        X_tra = prep_base(X_tra)
        X_val = prep_base(X_val)

        X_tra['tags'] = X_tra['tags'].fillna('0').apply(prep_tags)  # preprocess for user loop
        X_val['tags'] = X_val['tags'].fillna('0').apply(prep_tags)  # preprocess for user loop

        X_tra_wo_lec = prep_base(X_tra_wo_lec)
        X_val_wo_lec = prep_base(X_val_wo_lec)

        X_tra_wo_lec = pd.merge(X_tra_wo_lec, question[['question_id', 'lr', 'part', 'tags']], left_on='content_id', right_on='question_id', how='left')
        X_val_wo_lec = pd.merge(X_val_wo_lec, question[['question_id', 'lr', 'part', 'tags']], left_on='content_id', right_on='question_id', how='left')

        agg = add_modified_target_based_on_user_answer(X_tra_wo_lec, question)
        X_tra_wo_lec = pd.merge(X_tra_wo_lec, agg[['content_id', 'user_answer', 'weighted_answered_correctly', 'weighted_score']], on=['content_id', 'user_answer'], how='left')

        if os.path.exists(f'../save/et_table_{FOLD_NAME}.feather'):
            logging.info('skip make_elapsed_time_table')
            et_table = pd.read_feather(f'../save/et_table_{FOLD_NAME}.feather')
        else:
            et_table = make_elapsed_time_table(X_tra_wo_lec, FOLD_NAME)

        if os.path.exists(f'../save/rp_table_{FOLD_NAME}.feather'):
            logging.info('skip make_repeat_table')
            rp_table = pd.read_feather(f'../save/rp_table_{FOLD_NAME}.feather')
        else:
            rp_table = make_repeat_table(X_tra_wo_lec, FOLD_NAME)

        w2v_features = pd.read_feather(w2v_feature_path)
        g_features = pd.read_feather(g_feature_path)
        corr_g_features = pd.read_feather(corr_g_feature_path)
        ge_dw_features = pd.read_csv(ge_dw_feature_path)
        ge_s2v_features = pd.read_csv(ge_s2v_feature_path)

        graph_metrics_features = pd.DataFrame()
        graph_metrics_features['content_id'] = question['question_id']
        for graph_name in config.graphs:
            for metric_name in config.metrics:
                graph_metrics_features = pd.merge(graph_metrics_features, pd.read_feather(f'../save/{graph_name}_{metric_name}.feather'), on='content_id', how='left')

    del agg
    gc.collect()

    with t.timer('reduce Mem'):
        X_tra = reduce_mem_usage(X_tra)
        X_val = reduce_mem_usage(X_val)
        X_tra_wo_lec = reduce_mem_usage(X_tra_wo_lec)
        X_val_wo_lec = reduce_mem_usage(X_val_wo_lec)

    with t.timer('init FE Model'):
        riiidFE = riiidFE(FE_MODEL_PARAMS)
        riiidFE.set_use_features(use_features)
        riiidFE.set_train_mn(X_tra_wo_lec)
        riiidFE.set_repeat_mn(rp_table)
        riiidFE.set_cat_te_dict(X_tra_wo_lec, question)

    with t.timer('FE - content_id features'):
        riiidFE.extract_content_id_feat(
            X_tra_wo_lec, rp_table, et_table, question,
            w2v_features, g_features, corr_g_features, ge_dw_features
        )
        X_tra_wo_lec = pd.merge(X_tra_wo_lec, riiidFE.content_id_df, on='content_id', how='left')
        X_val_wo_lec = pd.merge(X_val_wo_lec, riiidFE.content_id_df, on='content_id', how='left')

    del w2v_features
    del g_features
    del corr_g_features
    del ge_dw_features
    del rp_table
    del et_table
    gc.collect()

    with t.timer('reduce Mem'):
        X_tra_wo_lec = reduce_mem_usage(X_tra_wo_lec)
        X_val_wo_lec = reduce_mem_usage(X_val_wo_lec)

    with t.timer('FE - loop features'):
        user_feat_df = riiidFE.add_user_feats(X_tra, add_feat=True, update_dict=True, val=False)
        del X_tra
        gc.collect()

        X_tra_wo_lec = pd.concat([X_tra_wo_lec, user_feat_df], axis=1)

        del user_feat_df
        gc.collect()
        X_tra_wo_lec = reduce_mem_usage(X_tra_wo_lec)

        # val only flag
        user_feat_df = riiidFE.add_user_feats(X_val, add_feat=True, update_dict=True, val=True)
        del X_val
        gc.collect()

        X_val_wo_lec = pd.concat([X_val_wo_lec, user_feat_df], axis=1)

        del user_feat_df
        gc.collect()
        X_val_wo_lec = reduce_mem_usage(X_val_wo_lec)

    if SAVE_EXTACTED_FEATURES is True:
        X_tra_wo_lec.to_feather(f'../save/{EXP_NAME}_{run_id}/X_tra_wo_lec.feather')
        X_val_wo_lec.to_feather(f'../save/{EXP_NAME}_{run_id}/X_val_wo_lec.feather')

    with t.timer('Save FE Model'):
        os.mkdir(f'../save/{EXP_NAME}_{run_id}/model')
        save_as_pkl(riiidFE, f'../save/{EXP_NAME}_{run_id}/features/riiidFE.pkl')

    with t.timer('Train Model'):

        model = catboost.CatBoostClassifier(
            **CAT_PARAMS
        )
        logging.info(f'Sample Num: {len(X_tra_wo_lec)}')
        logging.info(f'Feature Num: {len(use_features)}')
        print(len(use_features))

        model.fit(
            X_tra_wo_lec[use_features], X_tra_wo_lec['answered_correctly'].values,
            cat_features=[],
            eval_set=(X_val_wo_lec[use_features], X_val_wo_lec['answered_correctly'].values),
            use_best_model=True,
            verbose=100
        )

        preds_val = model.predict_proba(X_val_wo_lec[use_features])[:, 1]
        auc = roc_auc_score(X_val_wo_lec['answered_correctly'].values, preds_val)

    with t.timer('Feature importance + a'):
        imp = model.get_feature_importance()
        imp = pd.DataFrame(imp, index=use_features, columns=['importance']).sort_values('importance', ascending=False)

        imp = imp.reset_index()
        imp.columns = ['feat', 'importance']
        f_unique = X_tra_wo_lec[use_features].nunique().reset_index()
        f_unique.columns = ['feat', 'nunique']

        imp = pd.merge(imp, f_unique, on='feat')
        imp['min'] = X_tra_wo_lec[imp.feat.tolist()].min().values
        imp['max'] = X_tra_wo_lec[imp.feat.tolist()].max().values
        imp['mean'] = X_tra_wo_lec[imp.feat.tolist()].mean().values

    with t.timer('Save Model'):
        save_as_pkl(model, f'../save/{EXP_NAME}_{run_id}/model/cat_model.pkl')
        imp.to_csv(f'../save/{EXP_NAME}_{run_id}/importance_{run_id}_{auc}.csv')
        pd.DataFrame(preds_val, columns=['preds']).to_csv(f'../save/{EXP_NAME}_{run_id}/preds_val_{run_id}_{auc}.csv')
        save_mlflow(run_id, auc)
