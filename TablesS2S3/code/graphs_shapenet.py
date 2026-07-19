import numpy as np
import pandas as pd
import networkx as nx

np.random.seed(0)

def split_filename(x):
    parts = x.split('_')
    if len(parts) > 3:
        return parts[1] + parts[2]
    else:
        return parts[1]
    
# %%
def get_type_from_filename(filename, top_instance):
    """
    Extracts the 'type' (rank1 or other) from a filename.
    """

    fname = split_filename(filename)
    if fname == top_instance:
        return "Rank1"
    else:
        return "Other"
            
def get_key_with_highest_value(datasets, cats):
    instance_dir = {}
    
    for data in datasets:
        df = pd.read_csv(f"../../Fig3/data/shapenet/train_{data}.csv")
        df['type_recomputed'] = df['Filename'].apply(lambda x: split_filename(x))
        
        cat_dir = {}
        for cat in cats:
            df_cat = df [ df['Category'] == cat ].copy()
            data_dict = dict(df_cat['type_recomputed'].value_counts())
            cat_dir[cat] = max(data_dict, key=data_dict.get)
            
            del df_cat, data_dict

        instance_dir[data] = cat_dir

        del df,cat_dir 

    return instance_dir

def build_graph(df, threshold, instance_dir, data, cat):

    top_instance = instance_dir[data][cat]

    G = nx.Graph()
    nodes = set(df['instance_i']).union(set(df['instance_j'])) # can convert to list if order matters
    
    # sanity check
    assert len(nodes) == len(list(set(df['instance_i']))) + 1
    
    # add all nodes, irrespective of whether they will have edges or not
    for node in nodes:
        node_type = get_type_from_filename(node, top_instance) # type can be rank 1 or other
        G.add_node(node, type=node_type)
    
    # add nodes that have distance below threshold, and exclude self-comparisons
    for index, row in df.iterrows():
        node1 = row['instance_i']
        node2 = row['instance_j']
        distance = row['inv_distance']

        if (node1 == node2) or (distance > threshold):
            continue
        else:
            similarity = 1 - distance # due to euclidean distance
            G.add_edge(node1, node2, weight=distance, similarity=similarity)
    
    return G


def calculate_graph_statistics(G):
    stats = {}

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        stats['avg_degree'] = 'NA'
        stats['proportion_connected_nodes'] = 'NA'
        stats['avg_connectivity'] = 'NA'

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
        stats['avg_connectivity'] = weighted_sum_of_aconn / total_pairs_in_analyzed_components

    else:
        raise Exception("should not be here")

    return stats

def main():
    df = pd.read_csv("../data/shapenet/shapenet_sim_data.csv")

    ######################################################
    df['inv_distance'] = 1 - df['PE-Spatial-L_cos_sim']
    # df['inv_distance'] = 1 - df['hist_corr']
    # df['inv_distance'] = 1 - df['clip_cos_sim']
    # df['inv_distance'] = 1 - df['gist_cos_sim']
    ######################################################

    assert df['dataset'].nunique() == 2

    # compute quantiles
    all_datasets = [
        "nonLumpy",
        "lumpy",
    ]

    nonLumpy_datasets = [
        "nonLumpy",
    ]

    lumpy_datasets = [
        "lumpy",
    ]

    nonLumpy_vals = {
        'avg_degree': [],
        'proportion_connected_nodes': [],
        'avg_connectivity': []
    }
    lumpy_vals = {
        'avg_degree': [],
        'proportion_connected_nodes': [],
        'avg_connectivity': []
    }
    
    cats = ["airplane", "bottle", "car", "chair", "camera", "bed"]
    instance_dir = get_key_with_highest_value(all_datasets, cats)
    combined_to_write = []

    print(f"cats: {cats}")
    for cat in cats:
        print('-'*20)
        print(f"Beginning category {cat}")

        qs = []
        for data in all_datasets:
            singled_df = df[df["dataset"] == data].copy()
            cat_df = singled_df[singled_df["category"] == f"{cat}_{cat}"].copy() # within category only
            quanti = cat_df['inv_distance'].quantile(.1)
            qs.append(quanti)
            del singled_df, cat_df, quanti

        quanti = min(qs)
        graphs = []

        for data in all_datasets:
            singled_df = df[df["dataset"] == data].copy()
            cat_df = singled_df[singled_df["category"] == f"{cat}_{cat}"].copy()
            G = build_graph(cat_df, quanti, instance_dir, data, cat)
            graphs.append((G, data))

            del singled_df, cat_df, G

        lst_to_write = []

        for i, (G, data) in enumerate(graphs):
            stats_to_write = {}

            stats = calculate_graph_statistics(G)
            if data in nonLumpy_datasets:
                nonLumpy_vals['avg_degree'].append(stats['avg_degree'])
                nonLumpy_vals['proportion_connected_nodes'].append(stats['proportion_connected_nodes'])
                nonLumpy_vals['avg_connectivity'].append(stats['avg_connectivity'])
                stats["dataset_agg"] = "nonLumpy"
            
            elif data in lumpy_datasets:
                lumpy_vals['avg_degree'].append(stats['avg_degree'])
                lumpy_vals['proportion_connected_nodes'].append(stats['proportion_connected_nodes'])
                lumpy_vals['avg_connectivity'].append(stats['avg_connectivity'])
                stats["dataset_agg"] = "lumpy"
            
            else:
                raise NotImplementedError("should not be here.")
            
            # stats["subj"] = "Random"
            stats['category'] = cat
            try:
                lst_to_write.append(pd.DataFrame(stats, index=[0]))
            except:
                print(stats_to_write)
                raise Exception("something went wrong")

            del G, stats, stats_to_write

        df_to_write = pd.concat(lst_to_write, ignore_index=True)
        print(df_to_write)
        df_to_write.to_csv(f"../data/shapenet/subs/graph_array_10pct_{cat}_PE.csv", index=False)
        combined_to_write.append(df_to_write)

        del lst_to_write, df_to_write
    
    # combined_df = pd.concat(combined_to_write, ignore_index=True)
    # combined_df.to_csv(f"../data/shapenet/graph_array_10pct_Clip.csv", index=False)

if __name__ == "__main__":
    main()