import numpy as np
import pandas as pd
from tqdm import tqdm

# w2vc
import logging
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD

from utils import (
    save_as_pkl
)


def prep_tags(x):
    return [int(i) for i in x.split()]


def reduce_svd(features, n_components):
    dr_model = TruncatedSVD(n_components=n_components, random_state=46)
    features_dr = dr_model.fit_transform(features)
    return features_dr


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


INPUT_DIR = '../input'
NUM_WORKER = 18
MODEL_NAME = 'Word2vec_withlec'

w2v_epoch_num = 3

w2v_sg = 1
w2v_num_features = 20
w2v_min_word_count = 1
w2v_context = 50
svd_dim = 5

if __name__ == "__main__":

    train = pd.read_feather(f'{INPUT_DIR}/train_v2.feather')
    question = pd.read_csv(f'{INPUT_DIR}/questions.csv')

    question['part'] = question['part'].fillna(0).astype('int8')
    question['tags'] = question['tags'].fillna('0').apply(prep_tags)
    question2tags = {i[0]: i[1] for i in question[['question_id', 'tags']].values}

    train = pd.merge(train, question[['question_id', 'part', 'tags']], left_on='content_id', right_on='question_id', how='left')
    train['part'] = train['part'].fillna(0.0).astype('int8')
    train['tags'] = train['tags'].fillna('0')

    train = train[['row_id', 'content_type_id', 'user_id', 'content_id', 'part', 'tags', 'answered_correctly']]
    user_seq = []
    for uid, udf in tqdm(train.groupby('user_id')):
        cseq = []
        pseq = []
        tsqe = []
        lseq = []
        conbined_seq = []
        for cti, uid, cid, part, tags in udf[['content_type_id', 'user_id', 'content_id', 'part', 'tags']].values:
            if cti == 1:
                lseq.append(f'l{cid}')
                conbined_seq.append(f'l{cid}')
            else:
                cseq.append(f'c{cid}')
                pseq.append(f'p{part}')
                tsqe.extend([f't{i}' for i in tags])
                conbined_seq.append(f'c{cid}')
                conbined_seq.append(f'p{part}')
                conbined_seq.extend([f't{i}' for i in tags])
        user_seq.append(cseq + pseq + tsqe + lseq + conbined_seq)

    w2v_model = Word2Vec(size=w2v_num_features, sg=w2v_sg, workers=NUM_WORKER, window=w2v_context, min_count=w2v_min_word_count)
    w2v_model.build_vocab(user_seq)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    w2v_model.train(user_seq, total_examples=w2v_model.corpus_count, epochs=w2v_epoch_num)

    w2v_model.save(f'../save/{MODEL_NAME}_nf{w2v_num_features}_sg{w2v_sg}_win{w2v_context}_minc{w2v_min_word_count}.model')

    ct_vectors = {}
    for cid, tags in question2tags.items():
        vec = np.zeros(w2v_num_features)
        for t in tags:
            vec += w2v_model.wv[f't{t}'] / len(tags)
        ct_vectors[f'ct{cid}'] = vec
    vocab = [i for i in w2v_model.wv.vocab.keys()]

    idx = 0
    cname2idx = {}
    for i in vocab:
        cname2idx[i] = idx
        idx += 1
    for i in question2tags.keys():
        cname2idx[f'ct{i}'] = idx
        idx += 1
    idx2cname = {j: i for i, j in cname2idx.items()}

    n = len(cname2idx)
    simmat = np.zeros((n, n), dtype=np.float32)
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            if idx2cname[i][:2] == 'ct' and idx2cname[j][:2] == 'ct':
                simmat[i, j] = cos_sim(ct_vectors[idx2cname[i]], ct_vectors[idx2cname[j]])
            elif idx2cname[i][:2] == 'ct' and idx2cname[j][:2] != 'ct':
                simmat[i, j] = cos_sim(ct_vectors[idx2cname[i]], w2v_model.wv[idx2cname[j]])
            elif idx2cname[i][:2] != 'ct' and idx2cname[j][:2] == 'ct':
                simmat[i, j] = cos_sim(w2v_model.wv[idx2cname[i]], ct_vectors[idx2cname[j]])
            else:
                simmat[i, j] = w2v_model.wv.similarity(idx2cname[i], idx2cname[j])
    save_as_pkl(simmat, f'../save/simmat_{MODEL_NAME}_nf{w2v_num_features}_sg{w2v_sg}_win{w2v_context}_minc{w2v_min_word_count}.pkl')
    save_as_pkl(cname2idx, f'../save/cname2idx_{MODEL_NAME}_nf{w2v_num_features}_sg{w2v_sg}_win{w2v_context}_minc{w2v_min_word_count}.pkl')

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
    w2v_features.to_feather(f'../save//{MODEL_NAME}_nf{w2v_num_features}_sg{w2v_sg}_win{w2v_context}_minc{w2v_min_word_count}_svd{svd_dim}.csv')
