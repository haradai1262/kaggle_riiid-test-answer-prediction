import pandas as pd
import networkx as nx

import logging
from sklearn.decomposition import TruncatedSVD
from ge import (
    DeepWalk,
    Struc2Vec
)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

graph_name = 'ans_trans_mat'
# graph_name = 'ans_corr_trans_mat'
graph_path = f'../save/{graph_name}_edge_list.txt'
NUM_WORKER = 8
svd_dim = 5

dw_walk_length = 10
dw_num_walks = 100
dw_window_size = 5
dw_iter = 3

n2v_walk_length = 10
n2v_num_walks = 100
n2v_p = 0.25
n2v_q = 4
n2v_window_size = 5
n2v_iter = 3

sdne_hidden_size = [256, 128]
sdne_batch_size = 512
sdne_epochs = 40

s2v_walk_length = 10
s2v_num_walks = 100
s2v_window_size = 5
s2v_iter = 3


def reduce_svd(features, n_components):
    dr_model = TruncatedSVD(n_components=n_components, random_state=46)
    features_dr = dr_model.fit_transform(features)
    return features_dr


def save_embedding(embeddings, name):
    ge_features = pd.DataFrame([i for i in range(13523)], columns=['content_id'])
    rows = []
    for i in range(13523):
        rows.append(embeddings[str(i)])
    ge_features = pd.concat([
        ge_features, pd.DataFrame(rows).add_prefix(f'ge_{name}_')
    ], axis=1)
    ge_features.to_csv(f'../save/graph_embedding_{graph_name}_{name}.csv')


def save_embedding_svd(embeddings, name, svd_dim):
    ge_features = pd.DataFrame([i for i in range(13523)], columns=['content_id'])

    rows = []
    for i in range(13523):
        rows.append(embeddings[str(i)])

    features_dr = reduce_svd(pd.DataFrame(rows).values, svd_dim)

    ge_features = pd.concat([
        ge_features, pd.DataFrame(features_dr).add_prefix(f'ge_{name}_svd_')
    ], axis=1)
    ge_features.to_csv(f'../save/graph_embedding_{name}_{graph_name}_svd{svd_dim}.csv')


if __name__ == "__main__":

    G = nx.read_edgelist(graph_path, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    model = DeepWalk(G, walk_length=dw_walk_length, num_walks=dw_num_walks, workers=NUM_WORKER)  # init model
    model.train(window_size=dw_window_size, iter=dw_iter)  # train model
    embeddings = model.get_embeddings()  # get embedding vectors

    save_embedding(embeddings, name='dw')
    save_embedding_svd(embeddings, name='dw', svd_dim=svd_dim)

    model = model = Struc2Vec(G, walk_length=s2v_walk_length, num_walks=s2v_num_walks, workers=NUM_WORKER, verbose=40)  # init model
    model.train(window_size=s2v_window_size, iter=s2v_iter)  # train model
    embeddings = model.get_embeddings()  # get embedding vectors

    save_embedding(embeddings, name='s2v')
    save_embedding_svd(embeddings, name='s2v', svd_dim=svd_dim)
