import pandas as pd
import numpy as np
from tqdm import tqdm


def prep_tags(x):
    return [int(i) for i in x.split()]


def part2lr(x):
    if x in [1, 2, 3, 4]:
        return 0
    if x in [5, 6, 7]:
        return 1


def merge_questions_w_prep(df, question):
    df = pd.merge(df, question[['question_id', 'part', 'tags', 'lr']], left_on='content_id', right_on='question_id', how='left')
    df['part'] = df['part'].fillna(0.0).astype('int8')
    df['tags'] = df['tags'].fillna('0')
    return df


def prep_base(df):
    df['prior_question_elapsed_time'] = df['prior_question_elapsed_time'].fillna(-1).astype(int)
    df['prior_question_had_explanation'] = df['prior_question_had_explanation'].fillna(False).astype('int8')
    return df


def add_modified_target_based_on_user_answer(X_tra_wo_lec, question):

    # add modified target values 'weighted_answered_correctly', 'weighted_score'
    agg = X_tra_wo_lec.groupby(['content_id', 'user_answer'])['user_answer'].agg(['count']).reset_index()
    ccnt = X_tra_wo_lec['content_id'].value_counts().reset_index()
    ccnt.columns = ['content_id', 'c_count']

    agg = pd.merge(agg, ccnt, on='content_id', how='left')
    agg['choice_rate'] = agg['count'] / agg['c_count']
    agg = pd.merge(agg, question[['question_id', 'correct_answer']], right_on='question_id', left_on='content_id', how='left')
    agg['answered_correctly'] = (agg['user_answer'] == agg['correct_answer']) * 1

    # 'weighted_answered_correctly', 難しい問題に強い重み
    agg['weighted_answered_correctly'] = (1.0 - agg['answered_correctly'] * agg['choice_rate']) * agg['answered_correctly']

    # 'weighted_score', 不正解度合いによって重み付け
    agg2 = pd.DataFrame()
    for cid, tmp in tqdm(agg.groupby('content_id')):
        x = tmp[tmp.answered_correctly == 1]['choice_rate'].values.tolist() + tmp[tmp.answered_correctly != 1].sort_values('choice_rate', ascending=False)['choice_rate'].values.tolist()
        weighted_score = [1.0]
        for i in range(len(tmp) - 1):
            weighted_score.append(weighted_score[-1] - x[i])
        tmp = tmp.sort_values(by=['answered_correctly', 'choice_rate'], ascending=False)
        tmp['weighted_score'] = weighted_score
        agg2 = pd.concat([agg2, tmp])
    return agg2


def make_elapsed_time_table(X_tra_wo_lec, FOLD_NAME):
    rows = []
    for uid, udf in tqdm(X_tra_wo_lec[['user_id', 'timestamp', 'content_id', 'part', 'prior_question_had_explanation', 'prior_question_elapsed_time']].groupby('user_id')):
        udf['question_elapsed_time'] = udf['prior_question_elapsed_time'].shift(-1)
        udf = udf[~udf.question_elapsed_time.isna()]
        rows.extend(udf.values)

    et_table = pd.DataFrame(rows, columns=['user_id', 'timestamp', 'content_id', 'part', 'prior_question_had_explanation', 'prior_question_elapsed_time', 'question_elapsed_time'])
    et_table['content_id'] = et_table['content_id'].astype(int)
    et_table['part'] = et_table['part'].astype(int)
    et_table['prior_question_had_explanation'] = et_table['prior_question_had_explanation'].astype(int)
    et_table['question_elapsed_time'] = et_table['question_elapsed_time'].astype(int)
    et_table.to_feather(f'../save/et_table_{FOLD_NAME}.feather')
    return et_table


def make_repeat_table(X_tra_wo_lec, FOLD_NAME):
    repeat_idx = []
    question_u_dict = {}
    for r, uid, cid, ans in tqdm(X_tra_wo_lec[['row_id', 'user_id', 'content_id', 'answered_correctly']].values):
        if uid not in question_u_dict:
            question_u_dict[uid] = np.zeros(13523, dtype=np.uint8)
        if question_u_dict[uid][cid] > 0:
            repeat_idx.append(r)
        question_u_dict[uid][cid] += 1
    repeat = X_tra_wo_lec[X_tra_wo_lec.row_id.isin(repeat_idx)]
    repeat.reset_index(drop=True).to_feather(f'../save/repeat_{FOLD_NAME}.feather')
    return repeat
