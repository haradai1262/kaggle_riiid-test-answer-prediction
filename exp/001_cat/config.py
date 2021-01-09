# RANDOM_STATE = 46
RANDOM_STATE = 20201209

FOLD_NAME = 'vlatest_ALL_2p5M'
# FOLD_NAME = 'vlatest_10M_2p5M'


FE_MODEL_PARAMS = {
    'TE_smooth': 5,
    'ansrec_max_len': 120,
    'timestamprec_max_len': 120,
    'prev_question_len': 120,
    'sequence_te_smooth': 10,
    'use_previous_similarities_num': 9
}

CAT_PARAMS = {
    'learning_rate': 0.1,
    'depth': 8,
    'l2_leaf_reg': 3.0,
    'bagging_temperature': 0.2,
    'border_count': 128,
    'od_type': 'Iter',
    'metric_period': 50,
    'iterations': 30000,
    'od_wait': 50,
    'random_seed': RANDOM_STATE,
    'task_type': 'GPU',
    'gpu_ram_part': 0.99,
}

CAT_PARAMS['loss_function'] = 'Logloss'
CAT_PARAMS['eval_metric'] = 'AUC'

content_tsne_path = '../save/content_id__tsne.feather'
content_tsne_kmeans_path = '../save/content_id__tsne_kmeans.feather'
content_cow2v_kmeans_path = '../save/content_id__cow2v_kmeans.feather'

w2v_epoch_num = 3
w2v_num_features = 50
w2v_sg = 1
w2v_context = 50
w2v_min_word_count = 1
w2v_svd_dim = 5
w2v_feature_path = f'../save/Word2vec__nf{w2v_num_features}_sg{w2v_sg}_win{w2v_context}_minc{w2v_min_word_count}_svd{w2v_svd_dim}.feather'

g_svd_dim = 3
g_feature_path = f'../save/ans_trans_Graph_SVD_svd{g_svd_dim}.feather'
corr_g_feature_path = f'../save/ans_corr_trans_Graph_SVD_svd{g_svd_dim}.feather'

graph_name = 'ans_trans_mat'
name = 'dw'
ge_svd_dim = 5
ge_dw_feature_path = f'../save/graph_embedding_{name}_{graph_name}_svd{ge_svd_dim}.csv'

graph_name = 'ans_trans_mat'
name = 's2v'
ge_svd_dim = 5
ge_s2v_feature_path = f'../save/graph_embedding_{name}_{graph_name}_svd{ge_svd_dim}.csv'

base_feature = [
    'timestamp',
    # 'user_id',
    'content_id',
    'task_container_id',
    'prior_question_elapsed_time',
    # 'prior_question_had_explanation',
    # 'lr',
    # 'part',
]

part_agg_feature = [
    # 'part__count',
    # 'part__std',
    # 'part__answered_correctly_sm5',
    # 'part__weighted_answered_correctly_sm5',
    # 'part__weighted_score_sm5',
    # 'part__answered_correctly_sm5_repeat',
    # 'part__weighted_answered_correctly_sm5_repeat',
    # 'part__weighted_score_sm5_repeat',
    # 'part__question_elapsed_time_sm5',
    'part__svd_0',
    # 'part__svd_1',
    # 'part__svd_2',
    # 'part__svd_3',
    # 'part__svd_4',
]

content_agg_feature = [
    # 'content_id__count',
    # 'content_id__std',
    'content_id__count_repeat',
    'content_id__std_repeat',
    # 'content_id__unique_user',
    'content_id__answered_correctly_sm5',
    'content_id__weighted_answered_correctly_sm5',
    # 'content_id__weighted_score_sm5',
    'content_id__question_elapsed_time_sm5',
    'content_id__answered_correctly_sm5_repeat',
    'content_id__weighted_answered_correctly_sm5_repeat',
    'content_id__weighted_score_sm5_repeat',
    'content_id__entropy__user_answer',
    # 'content_id__entropy__answered_correctly',
    # 'content_id__svd_0',
    'content_id__svd_1',
    'content_id__svd_2',
    'content_id__svd_3',
    'content_id__svd_4',
    'content_id_tsne_0',
    'content_id_tsne_1',
]

content_session_feature = [
    'user_session_start__mean_smooth',
    'user_session_end__mean_smooth'
]

content_order_feature = [
    # 'content_order_mean',
    'content_order_median',
    # 'content_order_max',
    # 'content_order_min',
    'content_order_std',
]

tag_feature = [
    'tags_max',
    'tags_min',
    # 'tags_cnt',
    # 'tags_mean',
    # 'tags_std',
]

w2v_feature = [f'w2v_{i}' for i in [0, 1, 2, 3, 4]]
g_feature = [f'gsvd_{i}' for i in [2, 5]]
corr_g_feature = [f'corr_gsvd_{i}' for i in [3]]
ge_dw_feature = [f'ge_dw_svd_{i}' for i in [0, 1, 2, 3]]
ge_s2v_feature = [f'ge_s2v_svd_{i}' for i in range(ge_svd_dim)]

# w2v_feature = [f'w2v_{i}' for i in range(w2v_svd_dim)]
# g_feature = [f'gsvd_{i}' for i in range(g_svd_dim * 2)]
# corr_g_feature = [f'corr_gsvd_{i}' for i in range(g_svd_dim * 2)]
# ge_dw_feature = [f'ge_dw_svd_{i}' for i in range(ge_svd_dim)]
# ge_s2v_feature = [f'ge_s2v_svd_{i}' for i in range(ge_svd_dim)]

graphs = [
    'ans_trans_mat',
    # 'ans_corr_trans_mat'
]

metrics = [
    # 'degree_centrality',
    # 'in_degree_centrality',
    # 'out_degree_centrality',
    'eigenvector_centrality',
    # 'closeness_centrality',
    'betweenness_centrality',
    # 'harmonic_centrality',
    'trophic_levels',
]

content_graph_metric_features = []
for graph_name in graphs:
    for metric_name in metrics:
        content_graph_metric_features.append(f'{graph_name}_{metric_name}')

user_history_feature = [
    # 'u_cnt',
    # 'u_corr_cnt',
    'u_corr_rate_smooth',  # 'u_corr_rate',
    # 'u_latest1_corr_rate',
    # 'u_latest3_corr_rate',
    # 'u_latest5_corr_rate',
    'u_latest10_corr_rate',
    'u_latest30_corr_rate',
    # 'u_latest100_corr_rate',
    'u_in_seq_corr_rate',
    # 'u_latest10_corr_rate_div_all',
    # 'u_latest30_corr_rate_div_all',
    # 'u_latest100_corr_rate_div_all',
    'u_latest10_corr_rate_diff_all',
    'u_latest30_corr_rate_diff_all',
    # 'u_latest100_corr_rate_diff_all',
    # 'u_session_cnt',
    # 'u_session_change',
    # 'u_session_short_cnt',
    # 'u_session_short_change',
    # 'u_session_long_cnt',
    # 'u_session_long_change',
    # 'u_cnt_in_session',
    'u_cnt_in_session_short',
    # 'u_cnt_in_session_long',
    # 'u_cnt_density',
    # 'u_corr_cnt_density',
    # 'u_session_cnt_density',
    # 'u_session_short_cnt_density',
    # 'u_session_long_cnt_density',
    # 'u_cnt_in_session_density',
    'u_cnt_in_session_short_density',
    # 'u_cnt_in_session_long_density',
    # 'u_corr_rate_in_session',
    'u_corr_rate_in_session_short',
    # 'u_corr_rate_in_session_long',
    'seq2dec_w3',
    # 'seq2dec_w5',
    'seq2dec_w7',
    # 'seq2dec_w13',
    # 'seq2dec_w17',
    # 'seq2dec_w19',
    'content_id_diff_abs',
    'task_container_id_diff_abs',
    'u_first_content',
    'u_first10_corr_rate',
    'u_first1day_corr_rate',
    'u_first1week_corr_rate',
    'u_first1day_cnt',
    'u_first1week_cnt',
    'u_uans_rate',
    'u_incorr_uans_rate',
    'u_uans_rate_max_diff',
    'u_incorr_uans_rate_max_diff',
]

user_history_hade_feature = [
    # 'u_hade_cnt',
    # 'u_hade_corr_cnt',
    'u_hade_corr_rate',
    'u_hade_rate',
    'u_hade_div_corr_cnt',
]

user_history_question_feature = [
    # 'u_hist_ques_ansed',
    # 'u_hist_ques_corr_ansed',
    # 'u_hist_ques_num',
    # 'u_hist_ques_corr_num',
    # 'u_hist_ques_num_corr_rate',
    # 'u_hist_ques_nuni',
    # 'u_hist_ques_corr_nuni',
    # 'u_hist_ques_nuni_corr_rate',
    'u_hist_ques_in_seq',
    'u_hist_ques_in_seq_num',
    #
    # 'u_hist_queskm20_num',
    # 'u_hist_queskm20_corr_num',
    # 'u_hist_queskm20_corr_rate',
    'u_hist_queskm100_num',
    # 'u_hist_queskm100_corr_num',
    'u_hist_queskm100_corr_rate',
    # 'u_hist_queskm200_num',
    # 'u_hist_queskm200_corr_num',
    # 'u_hist_queskm200_corr_rate',
    #
    # 'u_hist_ques_cokm20_num',
    # 'u_hist_ques_cokm20_corr_num',
    # 'u_hist_ques_cokm20_corr_rate',
    'u_hist_ques_cokm100_num',
    # 'u_hist_ques_cokm100_corr_num',
    'u_hist_ques_cokm100_corr_rate',
    # 'u_hist_ques_cokm200_num',
    # 'u_hist_ques_cokm200_corr_num',
    # 'u_hist_ques_cokm200_corr_rate',
    #
    # 'u_hist_ques_incokm20_num',
    # 'u_hist_ques_incokm20_corr_num',
    # 'u_hist_ques_incokm20_corr_rate',
    'u_hist_ques_incokm100_num',
    # 'u_hist_ques_incokm100_corr_num',
    'u_hist_ques_incokm100_corr_rate',
    # 'u_hist_ques_incokm200_num',
    # 'u_hist_ques_incokm200_corr_num',
    # 'u_hist_ques_incokm200_corr_rate',
    #
    'u_hist_ques_sim_prevques',
    'u_hist_ques_sim_prevques_ans',
    # 'u_hist_part_sim_prevpart',
    # 'u_hist_part_sim_prevpart_ans',
    'u_hist_ques_sim_prevpart',
    'u_hist_ques_sim_prevpart_ans',
    'u_hist_part_sim_prevques',
    'u_hist_part_sim_prevques_ans',
    #
    'u_hist_ques_sim_prev2ques',
    'u_hist_ques_sim_prev2ques_ans',
    # 'u_hist_part_sim_prev2part',
    # 'u_hist_part_sim_prev2part_ans',
    # 'u_hist_ques_sim_prev2part',
    'u_hist_ques_sim_prev2part_ans',
    # 'u_hist_part_sim_prev2ques',
    'u_hist_part_sim_prev2ques_ans',
    #
    'u_hist_ques_sim_prev3ques',
    'u_hist_ques_sim_prev3ques_ans',
    #
    'u_hist_ques_sim_prev4ques',
    # 'u_hist_ques_sim_prev4ques_ans',
    #
    # 'u_hist_ques_sim_prev5ques',
    # 'u_hist_ques_sim_prev5ques_ans',
    #
    'u_hist_ques_sim_prevques_mean',
    'u_hist_ques_sim_prevques_max',
    'u_hist_ques_sim_prevques_min',
    # 'u_hist_part_sim_prevpart_mean',
    # 'u_hist_part_sim_prevpart_max',
    # 'u_hist_part_sim_prevpart_min',
    # 'u_hist_ques_sim_prevpart_mean',
    # 'u_hist_ques_sim_prevpart_max',
    # 'u_hist_ques_sim_prevpart_min',
    'u_hist_part_sim_prevques_mean',
    'u_hist_part_sim_prevques_max',
    'u_hist_part_sim_prevques_min',
    #
    'u_hist_ques_sim_prevques_incorr_mean',
    'u_hist_ques_sim_prevques_incorr_max',
    'u_hist_ques_sim_prevques_incorr_min',
    # 'u_hist_part_sim_prevpart_incorr_mean',
    # 'u_hist_part_sim_prevpart_incorr_max',
    # 'u_hist_part_sim_prevpart_incorr_min',
    # 'u_hist_ques_sim_prevpart_incorr_mean',
    # 'u_hist_ques_sim_prevpart_incorr_max',
    # 'u_hist_ques_sim_prevpart_incorr_min',
    'u_hist_part_sim_prevques_incorr_mean',
    'u_hist_part_sim_prevques_incorr_max',
    'u_hist_part_sim_prevques_incorr_min',
]


user_history_part_feature = [
    # 'u_hist_part_ansed',
    # 'u_hist_part_corr_ansed',
    'u_hist_part_num',
    'u_hist_part_corr_num',
    # 'u_hist_part_nuni',
    # 'u_hist_part_corr_nuni',
    'u_part_1_cnt',
    'u_part_2_cnt',
    'u_part_3_cnt',
    'u_part_4_cnt',
    'u_part_5_cnt',
    'u_part_6_cnt',
    'u_part_7_cnt',
    'u_part_1_corr_rate',
    'u_part_2_corr_rate',
    'u_part_3_corr_rate',
    'u_part_4_corr_rate',
    'u_part_5_corr_rate',
    'u_part_6_corr_rate',
    'u_part_7_corr_rate',
    # 'u_hist_part_nuni_corr_rate',
    'u_hist_part_num_corr_rate',
    # 'u_hist_part_std',
    # 'u_hist_part_corr_std',
    # 'u_hist_part_rank',
]

user_history_tag_feature = [
    # 'u_hist_tag_ansed',
    # 'u_hist_tag_corr_ansed',
    'u_hist_tag_num',
    # 'u_hist_tag_corr_num',
    # 'u_hist_tag_num_max',
    # 'u_hist_tag_corr_num_max',
    # 'u_hist_tag_num_min',
    'u_hist_tag_corr_num_min',
    # 'u_hist_tag_nuni',
    # 'u_hist_tag_corr_nuni',
    # 'u_hist_tag_max',
    # 'u_hist_tag_min',
    # 'u_hist_tag_sum',
    # 'u_hist_tag_std',
    # 'u_hist_tag_corr_max',
    # 'u_hist_tag_corr_min',
    # 'u_hist_tag_corr_sum',
    # 'u_hist_tag_corr_std',
    # 'u_hist_tag_nuni_corr_rate',
    'u_hist_tag_num_corr_rate',
    # 'u_hist_tag_rank_max',
    'u_hist_tag_rank_min',
    'u_hist_tag_rank_sum',
    # 'u_hist_tag_corr_rate_max',
    'u_hist_tag_corr_rate_min',
    'u_hist_tag_corr_rate_mn',
]

user_history_lecture_feature = [
    # 'u_hist_lec',
    # 'u_hist_lec_cnt',
    'u_hist_lec_part_cnt',
    # 'u_hist_lec_type_0_cnt',
    'u_hist_lec_type_1_cnt',
    'u_hist_lec_type_2_cnt',
    # 'u_hist_lec_type_3_cnt',
    'u_hist_ans_num_from_lec',
    'u_hist_sim_cid_latest_lec',
    'u_hist_sim_cid_latest_lec_part',
]

user_history_time_feature = [
    'u_prev_difftime',
    'u_prev_difftime_2',
    'u_prev_difftime_3',
    'u_prev_difftime_4',
    # 'u_prev_difftime_5',
    'u_prev_difftime_incorr',
    'u_prev_lagtime',
    'u_prev_lagtime_2',
    'u_prev_lagtime_3',
    # 'u_prev_lagtime_4',
    # 'u_hist_et_sum',
    'u_hist_et_sum_div_cnt',
    'u_hist_et_sum_div_corr_cnt',
    # 'u_hist_difftime_sum',
    # 'u_hist_lagtime_sum',
    'u_latest5_lagtime_max',
    # 'u_latest5_difftime_max',
    # 'u_latest5_lagtime_mn',
    # 'u_latest5_difftime_mn',
    'u_latest5_lagtime_median',
    'u_latest5_difftime_median',
    # 'u_latest10_lagtime_max',
    # 'u_latest10_difftime_max',
    # 'u_latest10_lagtime_mn',
    # 'u_latest10_difftime_mn',
    'u_latest10_lagtime_median',
    'u_latest10_difftime_median',
    'u_latest5_et_mn',
    # 'u_latest5_et_max',
    # 'u_latest5_et_median',
    'u_latest10_et_mn',
    'u_latest10_et_max',
    # 'u_latest10_et_median',
    # 'u_latest100_lagtime_max',
    # 'u_latest100_difftime_max',
    # 'u_latest100_lagtime_mn',
    # 'u_latest100_difftime_mn',
    # 'u_latest100_lagtime_median',
    # 'u_latest100_difftime_median',
    # 'u_latest100_et_median',
    'u_in_seq_et_max',
    # 'u_latest10_lagtime_corr_max',
    # 'u_latest10_lagtime_corr_mn',
    # 'u_latest10_difftime_corr_max',
    # 'u_latest10_difftime_corr_mn',
    # 'u_latest10_et_corr_max',
    # 'u_latest10_et_corr_mn',
    # 'u_prev_difftime_norm',
    # 'u_prev_lagtime_norm',
    # 'u_latest5_difftime_mn_norm',
    'u_prev_et_diff_time',
    # 'u_prev_et_lag_time',
    'u_prev_diff_lag_time',
    'u_prev_lag_diff_time',
    # 'u_ts_in_session',
    # 'u_ts_in_session_short',
    # 'u_ts_in_session_long',
    # 'timestamp_lag_div_rolling5_median_each_user',
    # 'timestamp_lag_div_rolling7_median_each_user',
    # 'timestamp_lag_div_rolling10_median_each_user',
    # 'timestamp_lag_diff_rolling5_median_each_user',
    # 'timestamp_lag_diff_rolling7_median_each_user',
    # 'timestamp_lag_diff_rolling10_median_each_user',
    # 'timestamp_lag_diff_median_each_user',
    # 'timestamp_lag_div_median_each_user',
    # 'timestamp_lag_median',
    # 'difftime_diff_latest100_difftime_median',
    'difftime_div_in_seq_difftime_median',
    # 'lagtime_diff_latest100_lagtime_median',
    # 'lagtime_div_latest100_lagtime_median',
    'difftime_diff_in_seq_lagtime_median',
    'difftime_div_in_seq_lagtime_median',
    # 'lagtime_diff_latest100_difftime_median',
    # 'lagtime_div_latest100_difftime_median',
]

use_features = []
use_features += base_feature
use_features += part_agg_feature
use_features += content_agg_feature
use_features += content_session_feature
use_features += content_order_feature
use_features += tag_feature
use_features += w2v_feature
use_features += g_feature
use_features += corr_g_feature
use_features += ge_dw_feature
use_features += ge_s2v_feature
use_features += content_graph_metric_features
use_features += user_history_feature
use_features += user_history_time_feature
use_features += user_history_hade_feature
use_features += user_history_question_feature
use_features += user_history_part_feature
use_features += user_history_tag_feature
use_features += user_history_lecture_feature
