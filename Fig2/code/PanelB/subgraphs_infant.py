
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.patches as mpatches
np.random.seed(0)
import json
import os
    
def get_type_from_filename(filename, top_instance):
    if filename in top_instance:
        return "Rank1"
    else:
        return "Other"           

def build_graph(df, threshold, instance_dir, cat):
    top_instance = set(instance_dir['top'][cat])
    all_instance = set(instance_dir['acceptable_imgs'][cat])

    G = nx.Graph()

    # get unique nodes
    nodes = set(df['instance_i']).union(set(df['instance_j'])) # can convert to list if order matters
    if len(nodes) < 10:
        return G, G

    assert len(nodes) == len(list(set(df['instance_i']))) + 1
    
    # add all nodes, irrespective of whether they will have edges or not
    for node in nodes:
        if node not in all_instance:
            continue

        node_type = get_type_from_filename(node, top_instance)
        G.add_node(node, type=node_type)
    
    # add nodes that have distance below threshold, and exclude self-comparisons
    for index, row in df.iterrows():
        node1 = row['instance_i']
        node2 = row['instance_j']
        distance = row['distance']

        if (node1 == node2) or (distance > threshold) or (node1 not in all_instance) or (node2 not in all_instance):
            continue
    
        similarity = 1 / distance
        G.add_edge(node1, node2, weight=distance, similarity=similarity)

    return G, None

def node_induced_subg(G, nd_type):
    assert type(nd_type) == str
    
    nodes_list = [node for node, data in G.nodes(data=True) if data.get('type') == nd_type]
    subG =  G.subgraph(nodes_list).copy()
    
    assert (subG.number_of_nodes() == 0) or all(data.get('type') == nd_type for _, data in subG.nodes(data=True))

    return subG

def edge_induced_subg(G, nd_type1, nd_type2):
    assert type(nd_type1) == str and type(nd_type2) == str
    
    edges_list = [
        (u, v) for u, v in G.edges()
        if (G.nodes[u].get('type') == nd_type1 and G.nodes[v].get('type') == nd_type2) or 
        (G.nodes[u].get('type') == nd_type2 and G.nodes[v].get('type') == nd_type1)
    ]

    subG = G.edge_subgraph(edges_list).copy()

    assert (G.number_of_nodes() == 0 and G.number_of_edges() == 0) or all({subG.nodes[u].get('type'), subG.nodes[v].get('type')} == {nd_type1, nd_type2} for u, v in subG.edges)
    
    return subG

def calculate_graph_statistics(G_orig):
    
    node_induced = True
    print(f"Node induced: {node_induced}")

    if node_induced:
        G = node_induced_subg(G=G_orig, nd_type='Other')
    else:
        G = edge_induced_subg(G=G_orig, nd_type1='Rank1', nd_type2='Other')
    
    del G_orig

    stats = {}

    if G.number_of_nodes() < 10 or G.number_of_edges() == 0:
        stats['avg_degree'] = 'NA'
        stats['proportion_connected_nodes'] = 'NA'
        stats['overall_avg_connectivity'] = 'NA'

        return stats

    # avg degree; either way
    # degrees = [d for n, d in G.degree()]
    #sum(degrees) / len(degrees)
    stats['avg_degree'] = 2*G.number_of_edges() / G.number_of_nodes()
    

    # components = [G.subgraph(c).copy() for c in nx.connected_components(G)] # if you want altogether
    single_node_components = [G.subgraph(c).copy() for c in nx.connected_components(G) if len(c) == 1]
    components = [G.subgraph(c).copy() for c in nx.connected_components(G) if len(c) > 1]

    stats['proportion_connected_nodes'] =  ( G.number_of_nodes() - len(single_node_components) ) / G.number_of_nodes()

    weighted_sum_of_aconn = 0
    total_pairs_in_analyzed_components = 0
    
    for i,component in enumerate(components):
        num_nodes = component.number_of_nodes()
        if num_nodes > 1: # metrics for components with at least 2 nodes
            try:
                num_pairs = num_nodes * (num_nodes - 1)
                aconnectivity = nx.average_node_connectivity(component)

                weighted_sum_of_aconn += aconnectivity * num_pairs

                total_pairs_in_analyzed_components += num_pairs       

            except:
                print(f"something went wrong")
        else:
            raise Exception("should have filtered out single node components")
    
    if total_pairs_in_analyzed_components > 0:
        stats['overall_avg_connectivity'] = weighted_sum_of_aconn / total_pairs_in_analyzed_components

    else:
        raise Exception("should not be here")

    return stats


# %%
def main():

    cats = ["bottle", "bowl", "chair", "cup", 
                    "door", "spoon", "table", "window"]
    path_to_csvs = "../../data/PanelA/deep_features/PE/PE-Spatial-L"
    file_pattern = "all2all_corr_bysubj_forRplots"

    df_lst = []
    column_names = ["subj", "instance_i", "instance_j", "category", "valid", "hist_corr", "cos_sim"]

    for cat in cats:
        temp_path = os.path.join(path_to_csvs, f"{file_pattern}_{cat}_withPE-Spatial-L-CLS.csv")
        df_temp = pd.read_csv(temp_path, header=None)
        df_temp.columns = column_names
        df_lst.append(df_temp)

        del df_temp, temp_path

    df_allsubjs = pd.concat(df_lst, ignore_index=True)
    
    print(f"[Before] {df_allsubjs.shape}")
    dfff = df_allsubjs [ df_allsubjs["valid"] == True ].copy()
    print(f"[After] {df_allsubjs.shape}")
    
    ######################################################
    dfff['distance'] = 1 - dfff['cos_sim']              # Initially, our analyses where based on euclidean distance; now that we use similarity, we just used distance as 1 - similarity;
    ######################################################

    subjs = list( dfff['subj'].unique() )
    print(subjs)

    # get rank 1 instance filenames to color graph nodes
    with open('../../instance_filenames.json', 'r') as f:
        instance_dir = json.load(f)
    
    lst_to_write = []
    for subj in subjs:

        instance_dir_subj = instance_dir[subj]#["top"]

        for i, cat in enumerate(cats):

            # filter by cat
            cat_df = dfff[
                (dfff["category"] == f"{cat}_{cat}") & \
                (dfff["subj"] == f"{subj}") 
            ].copy()
            
            quanti = cat_df['distance'].quantile(.1)
            
            print(f"{subj} --- {cat} --- {cat_df.shape}")
            
            # dont mix subjects
            assert cat_df['subj'].nunique() <= 1
            
            G, _ = build_graph(cat_df, quanti, instance_dir_subj, cat)
            
            np.random.seed(0)
            
            # get stats
            stats = calculate_graph_statistics(G)
            stats['subj'] = subj
            stats['category'] = cat
            
            try:
                lst_to_write.append(pd.DataFrame(stats, index=[0]))
            except:
                print(stats)
                raise Exception("should not be here")

            del G, stats, cat_df
    
    df_to_write = pd.concat(lst_to_write, ignore_index=True)
    df_to_write.to_csv(f"../../data/PanelC/PE_graphs/SUBgraphs_PE_10pct_nodeInd_O.csv", index=False)

if __name__ == "__main__":
    main()