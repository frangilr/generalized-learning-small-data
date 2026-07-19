
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.patches as mpatches
np.random.seed(0)
import json
import os

CONFIGS = {
    "RGB": {
        "real_dir": "../../data/PanelA/RGB_hist",
        "null_dir": "../../data/PanelA/RGB_hist",
        "real_suffix": "",
        "col_idx": "hist_corr",
        "columns": ["subj", "instance_i", "instance_j", "category", "valid", "hist_corr"],
        "out_dir": "../../data/PanelC/RGB_graphs",
        "feature_token": "RGB",
    },
    "Clip": {
        "real_dir": "../../data/PanelA/deep_features/Clip",
        "null_dir": "../../data/PanelA/deep_features/Clip",
        "real_suffix": "_withClip",
        "col_idx": "cos_sim",
        "columns": ["subj", "instance_i", "instance_j", "category", "valid", "hist_corr", "cos_sim"],
        "out_dir": "../../data/PanelC/Clip_graphs",
        "feature_token": "Clip",
    },
    "GIST": {
        "real_dir": "../../data/PanelA/deep_features/GIST_cosSim",
        "null_dir": "../../data/PanelA/deep_features/GIST_cosSim",
        "real_suffix": "_withGIST_cosSim",
        "col_idx": "cos_sim",
        "columns": ["subj", "instance_i", "instance_j", "category", "valid", "hist_corr", "cos_sim"],
        "out_dir": "../../data/PanelC/GIST_cosSim_graphs",
        "feature_token": "GIST_cosSim",
    },
    "PE": {
        "real_dir": "../../data/PanelA/deep_features/PE/PE-Spatial-L",
        "null_dir": "../../data/PanelA/deep_features/PE/PE-Spatial-L",
        "real_suffix": "_withPE-Spatial-L-CLS",
        "col_idx": "cos_sim",
        "columns": ["subj", "instance_i", "instance_j", "category", "valid", "hist_corr", "cos_sim"],
        "out_dir": "../../data/PanelC/PE_graphs",
        "feature_token": "PE",
    },
}

def get_type_from_filename(filename, top_instance):
    if filename in top_instance:
        return "Rank1"
    else:
        return "Other"           

def build_NULL_graph(num_nodes, threshold, seed=42):
    G = nx.random_geometric_graph(num_nodes, radius=threshold, seed=seed)
    nx.set_node_attributes(G, "Other", "type")

    pos = nx.get_node_attributes(G, "pos")

    # compute distances from stored positions and set attributes
    for u, v in G.edges():
        pu = np.asarray(pos[u], dtype=float)
        pv = np.asarray(pos[v], dtype=float)
        dist = float(np.linalg.norm(pu - pv))
        G[u][v]["distance"] = dist
        G[u][v]["weight"] = dist              # keep weight as distance
        G[u][v]["similarity"] = 1.0 - dist

    return G

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

    G_null = build_NULL_graph(G.number_of_nodes(), threshold)

    return G, G_null

def calculate_graph_statistics(G):
    stats = {}

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
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

def visualize_graph(G, title="k-NN Graph", node_size=2, ax=None):

    type_color_map = {
        'Rank1': "#FC8E62CF",
        'Other': "#8DA0CBEE"
    }
    node_colors_list = []
    
    pos = nx.spring_layout(G, seed=42, weight='similarity', k=1.)
    
    # get top or non top for for coloring
    for node_id in G.nodes():
        node_attributes = G.nodes[node_id]
        node_type = node_attributes.get('type', 'Unknown')
        if node_type not in type_color_map: print(node_type)
        node_colors_list.append(type_color_map.get(node_type, "error"))

    # Edges
    edge_weights = nx.get_edge_attributes(G, 'similarity')
    max_weight = max(edge_weights.values())

    edge_widths = [w / max_weight for w in edge_weights.values()]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    sc = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_size, node_color=node_colors_list,
        alpha=0.8
    )

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=1.) # NOTE the opacity of edges depends on the save format (e.g. PNG, PDF, SVG)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    legend_patches = []

    # sort for consistent legend order
    sorted_types_for_legend = ['Rank1', 'Other']

    for node_type in sorted_types_for_legend:
        type_present_in_graph = any(
            data.get('type', 'Unknown') == node_type for _, data in G.nodes(data=True)
        )
        if type_present_in_graph:
            color = type_color_map[node_type]
            legend_patches.append(mpatches.Patch(color=color, label=str(node_type)))
        else:
            pass
            # raise Exception("type should be in graph")
    
    if legend_patches:
        ax.legend(handles=legend_patches, title="Instance Type", loc="lower left", 
                   fontsize=10, title_fontsize=10, 
                  frameon=True, facecolor='white', framealpha=0.5)
    else:
        raise Exception("should not be here")

    return sc 


# %%
def main():

    cats = ["bottle", "bowl", "chair", "cup", 
                    "door", "spoon", "table", "window"]
    FEATURE = "PE"  # options: RGB, Clip, GIST, PE
    THRESHOLD_QUANTILE = .1
    PCT_LABEL = f"{int(THRESHOLD_QUANTILE * 100):02d}pct"
    config = CONFIGS[FEATURE]

    file_pattern = "all2all_corr_bysubj_forRplots"

    df_lst = []
    column_names = config["columns"]

    for cat in cats:
        temp_path = os.path.join(config["real_dir"], f"{file_pattern}_{cat}{config['real_suffix']}.csv")
        df_temp = pd.read_csv(temp_path, header=None)
        df_temp.columns = column_names
        df_lst.append(df_temp)

        del df_temp, temp_path

    df_allsubjs = pd.concat(df_lst, ignore_index=True)
    
    # print(f"[Before] {df_allsubjs.shape}")
    dfff = df_allsubjs [ df_allsubjs["valid"] == True ].copy()
    # print(f"[After] {df_allsubjs.shape}")
    
    dfff['distance'] = 1 - dfff[config["col_idx"]] 

    subjs = list( dfff['subj'].unique() )
    print(subjs)

    # get rank 1 instance filenames to color graph nodes
    with open('../../instance_filenames.json', 'r') as f:
        instance_dir = json.load(f)
    
    lst_to_write = []

    null = False
    word = "null" if null else "infant"

    for subj in subjs:

        instance_dir_subj = instance_dir[subj]

        for i, cat in enumerate(cats):
            curr_lst = []

            # filter by cat
            cat_df = dfff[
                (dfff["category"] == f"{cat}_{cat}") & \
                (dfff["subj"] == f"{subj}") 
            ].copy()
            
            quanti = cat_df['distance'].quantile(THRESHOLD_QUANTILE)
            
            print(f"{subj} --- {cat} --- {cat_df.shape}")
            
            # dont mix subjects
            assert cat_df['subj'].nunique() <= 1
            
            # first graph is real, second is null
            G0, G1 = build_graph(cat_df, quanti, instance_dir_subj, cat)

            G = G0 if not null else G1
            
            np.random.seed(0)
            
            # get stats
            stats = calculate_graph_statistics(G)
            stats['subj'] = subj
            stats['category'] = cat
            
            try:
                lst_to_write.append(pd.DataFrame(stats, index=[0]))
                curr_lst.append(pd.DataFrame(stats, index=[0]))
            except:
                print(stats)
                raise Exception("should not be here")

            curr_df = pd.concat(curr_lst, ignore_index=True)
            curr_df.to_csv(
                f"{config['out_dir']}/subs/graphs_{word}_{config['feature_token']}_{PCT_LABEL}_{subj}_{cat}.csv",
                index=False
            )

            del G, stats, cat_df, curr_lst, curr_df
    
    # df_to_write = pd.concat(lst_to_write, ignore_index=True)
    # df_to_write.to_csv(f"../../data/PanelC/GIST_graphs/graphs_infant_GIST_10pct.csv", index=False)

if __name__ == "__main__":
    main()
