import pandas as pd
import networkx as nx

graph_name = 'ans_trans_mat'
# graph_name = 'ans_corr_trans_mat'
graph_path = f'../save/{graph_name}_edge_list.txt'


def get_g_metrics(item_dict, name):
    rows = []
    for i, j in item_dict:
        rows.append([i, j])
    df = pd.DataFrame(rows, columns=['content_id', name])
    df['content_id'] = df['content_id'].astype(int)
    df = df.sort_values('content_id').reset_index(drop=True)
    return df


def extract_g_metrics(G, name):

    if name == 'degree_centrality':
        metric = nx.degree_centrality(G).items()

    if name == 'in_degree_centrality':
        metric = nx.in_degree_centrality(G).items()

    if name == 'out_degree_centrality':
        metric = nx.out_degree_centrality(G).items()

    if name == 'eigenvector_centrality':
        metric = nx.eigenvector_centrality(G).items()

    if name == 'closeness_centrality':
        metric = nx.closeness_centrality(G).items()

    if name == 'betweenness_centrality':
        metric = nx.betweenness_centrality(G).items()

    if name == 'harmonic_centrality':
        metric = nx.harmonic_centrality(G).items()

    if name == 'trophic_levels':
        metric = nx.trophic_levels(G).items()

    fname = f'{graph_name}_{name}'
    get_g_metrics(metric, name=fname)[['content_id', fname]].to_feather(f'../save/{fname}.feather')
    return


if __name__ == "__main__":

    G = nx.read_edgelist(graph_path, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    gdf = pd.DataFrame([i for i in range(13524)], columns=['content_id'])

    extract_g_metrics(G, name='degree_centrality')
    extract_g_metrics(G, name='in_degree_centrality')
    extract_g_metrics(G, name='out_degree_centrality')
    extract_g_metrics(G, name='eigenvector_centrality')
    extract_g_metrics(G, name='closeness_centrality')
    extract_g_metrics(G, name='betweenness_centrality')
    extract_g_metrics(G, name='harmonic_centrality')
    extract_g_metrics(G, name='trophic_levels')