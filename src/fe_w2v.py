import numpy as np
import pandas as pd
from tqdm import tqdm

# w2vc
import logging
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD


def prep_tags(x):
    return [int(i) for i in x.split()]


def reduce_svd(features, n_components):
    dr_model = TruncatedSVD(n_components=n_components, random_state=46)
    features_dr = dr_model.fit_transform(features)
    return features_dr


INPUT_DIR = '../input'
NUM_WORKER = 14

w2v_epoch_num = 5

w2v_sg = 1
w2v_num_features = 50
w2v_min_word_count = 1
w2v_context = 50
svd_dim = 5

if __name__ == "__main__":

    train = pd.read_feather(f'{INPUT_DIR}/train_v2.feather')
    question = pd.read_csv(f'{INPUT_DIR}/questions.csv')

    question['part'] = question['part'].fillna(0).astype('int8')
    question['tags'] = question['tags'].fillna('0').apply(prep_tags)

    train = pd.merge(train, question[['question_id', 'part', 'tags']], left_on='content_id', right_on='question_id', how='left')
    train['part'] = train['part'].fillna(0.0).astype('int8')
    train['tags'] = train['tags'].fillna('0')

    train = train[train.content_type_id == 0][['row_id', 'user_id', 'content_id', 'part', 'tags', 'answered_correctly']]
    user_seq = []
    for uid, udf in tqdm(train.groupby('user_id')):
        cseq = [f'c{i}' for i in udf['content_id'].values.tolist()]
        pseq = [f'p{i}' for i in udf['part'].values.tolist()]
        tags = []
        for i in udf['tags'].values:
            tags.append([f't{t}' for t in i])
        tsqe = []
        for i in tags:
            tsqe.extend(i)
        conbined_seq = []
        for i in range(len(udf)):
            conbined_seq.append(cseq[i])
            conbined_seq.append(pseq[i])
            conbined_seq.extend(tags[i])
        user_seq.append(
            cseq + pseq + tsqe + conbined_seq
        )

    w2v_model = Word2Vec(size=w2v_num_features, sg=w2v_sg, workers=NUM_WORKER, window=w2v_context, min_count=w2v_min_word_count)
    w2v_model.build_vocab(user_seq)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    w2v_model.train(user_seq, total_examples=w2v_model.corpus_count, epochs=w2v_epoch_num)

    w2v_model.save(f'../save/Word2vec__nf{w2v_num_features}_sg{w2v_sg}_win{w2v_context}_minc{w2v_min_word_count}.model')

    w2v_model = Word2Vec.load(f'../save/Word2vec__nf{w2v_num_features}_sg{w2v_sg}_win{w2v_context}_minc{w2v_min_word_count}.model')
    cids = []
    features = np.zeros((len(question), w2v_num_features * 4), dtype='float32')
    i = 0
    for cid, part, tags in tqdm(question[['question_id', 'part', 'tags']].values):
        c_vec = w2v_model[f'c{cid}']
        p_vec = w2v_model[f'p{part}']
        tag_vec = np.zeros(w2v_num_features, dtype='float32')
        for t in tags:
            tag_vec += w2v_model[f't{t}']
        tag_vec /= len(tags)
        mean_vec = (c_vec + p_vec + tag_vec) / 3
        features[i] = np.hstack([c_vec, p_vec, tag_vec, mean_vec])
        cids.append(cid)
        i += 1

    features_dr = reduce_svd(features, svd_dim)
    w2v_features = pd.DataFrame(cids, columns=['content_id'])
    w2v_features = pd.concat([
        w2v_features, pd.DataFrame(features_dr).add_prefix('w2v_')
    ], axis=1)
    w2v_features.to_feather(f'../save/Word2vec__nf{w2v_num_features}_sg{w2v_sg}_win{w2v_context}_minc{w2v_min_word_count}_svd{svd_dim}.csv')
