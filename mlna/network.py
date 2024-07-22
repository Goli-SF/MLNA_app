import networkx as nx
import pandas as pd
from preproc import extract_entities
import community.community_louvain as community_louvain
from pyvis.network import Network



def update_weights(df):
    """
    df is a pandas dataframe with the colums ['text_id', 'source', 'target'], storing network data. This function adds
    the 'weight' column to the df and updates the weights of each edge in the network data.
    """
    df['weight']=0
    op_df_1= pd.DataFrame(columns=['source', 'target', 'weight'])
    for i, row in df.iterrows():
        match = ((op_df_1['source'] == row['source']) & (op_df_1['target'] == row['target']))
        if not match.any():
            new_row = pd.DataFrame({'source': [row['source']], 'target': [row['target']], 'weight': [1]})
            op_df_1 = pd.concat([op_df_1, new_row], ignore_index=True)
        else:
            op_df_1.loc[match, 'weight'] += 1

    for j, row_1 in df.iterrows():
        for k, row_2 in op_df_1.iterrows():
            if row_1['source'] == row_2['source'] and row_1['target'] == row_2['target']:
                # Use loc to update the weight in the original DataFrame
                df.loc[j, 'weight'] = row_2['weight']

    return df



def get_network_data (text_df, entity_tags, user_ents=None, user_dict=None):
    """
    text_df is a dataframe with at least the columns 'text_id' and 'full_text'. This function extrats the desired
    entities (stored in entity_tags and user_ents) from every single full_text in the text_df and returns a dataframe
    consisting of sources, targes, weights and text_ids, ready for network analysis.
    """
    network_df = pd.DataFrame(columns=['text_id', 'source', 'target'])

    for i, row in text_df.iterrows():
        text= row['full_text']
        text_id= row['text_id']
        ent_dict= extract_entities(text, text_id, entity_tags, user_ents, user_dict)
        final_sources=[]
        final_targets=[]
        final_weights=[]
        for value in ent_dict['entities']:
            if len(value)>1:
                source= value[0]
                targets= value[1:]
                for target in targets:
                    final_sources.append(source)
                    final_targets.append(target)
                    final_weights.append(1)
        net_df= pd.DataFrame({'text_id': text_id,
                              'source': final_sources,
                              'target': final_targets,
                              'weight': final_weights
                              })
        network_df= pd.concat ([network_df, net_df], axis= 0).reset_index(drop=True)

        network_df= update_weights(network_df)

    return network_df



def detect_community (text_df, entity_tags, user_ents=None, user_dict=None, title='community_detection',\
    figsize=(1000, 700), bgcolor='black', font_color='white'):
    """
    Detects communities in the given texts within the text_df. It takes a list of entity tags, a string as title and a
    tuple for the figsize in the format (width, height). The user can also change the default background and font colors.
    The output is saved as an .html file onto the local drive.
    """
    network_df= get_network_data (text_df, entity_tags, user_ents, user_dict)
    G= nx.from_pandas_edgelist(network_df, source= "source", target= "target")

    for index, row in network_df.iterrows():
        G[row['source']][row['target']]['weight'] = row['weight']

    communities= community_louvain.best_partition(G)
    node_degree = dict(G.degree)

    combined_attributes= {}
    for node, community in communities.items():
        combined_attributes[node]= {
            'group': community,
            'size': node_degree[node]
        }

    nx.set_node_attributes(G, combined_attributes)

    com_net= Network(notebook=True, width=f'{figsize[0]}px', height=f'{figsize[1]}px',
                     bgcolor=bgcolor, font_color=font_color, cdn_resources='in_line')
    com_net.from_nx(G)
    com_net.save_graph(f'{title}.html')

    html_content= com_net.generate_html()

    return html_content



def visualize_network (text_df, entity_tags, user_ents=None, user_dict=None, core=False, select_nodes=None, sources=None,\
    title='network_visualization', figsize=(1000, 700), bgcolor='black', font_color='white'):
    """
    Extracts network data from text_df. The *args and **kwargs are as followes:

    * entity_tags: List of spaCy entitiy tags entered by the user.
    * user_ents: List of words that the user wants to included in the network as nodes.
    * user_dict: user dictionary
    * core: if True, the output of the function would be a core network visualization. If False, the function will
    visualize the whole network.
    * select_nodes: List of nodes that the user wants to see in the network.
    * sources: List of text_ids if the user only wants to see network relations between the nodes in one or more texts.
    * title: title of the .html file that stores the visualizaion.
    * figsize: size of the visualized network in pixels.
    * bgcolor: background color
    * font_color: font color

    The resulting network visualization is stored in an .html file in the working directory.
    """
    network_df= get_network_data (text_df, entity_tags, user_ents, user_dict)
    if select_nodes:
        for i, row in network_df.iterrows():
            if row['source'].lower() not in list(map(lambda x: x.lower(), select_nodes)) and\
            row['target'].lower() not in list(map(lambda x: x.lower(), select_nodes)):
                network_df= network_df.drop(i)
    if sources:
        for i, row in network_df.iterrows():
            if row['text_id'] not in sources:
                network_df= network_df.drop(i)

    G= nx.from_pandas_edgelist(network_df, source= "source", target= "target")

    for index, row in network_df.iterrows():
        G[row['source']][row['target']]['weight'] = row['weight']

    net= Network(notebook=True, width=f'{figsize[0]}px', height=f'{figsize[1]}px',
                 bgcolor=bgcolor, font_color=font_color, cdn_resources='in_line')

    node_degree=dict(G.degree)
    nx.set_node_attributes(G, node_degree, 'size')

    if core:
        net.from_nx(nx.k_core(G))
        net.save_graph(f'{title}.html')
    else:
        net.from_nx(G)
        net.save_graph(f'{title}.html')

    html_content= net.generate_html()

    return html_content



def filter_network_data (text_df, select_nodes, entity_tags, user_ents=None, user_dict=None, operator='OR'):
    """
    Applies a boolean mask to network_df and filters out only the edges with the nodes that the user has selected and
    saved in select_nodes as a list. If the operator equals "AND", select_nodes should contain only two nodes and the
    functions filters out all of the edges containing only the two desired nodes. If the operator equals "OR",
    select_nodes can be a longer list and the functions filters out all of the edges that cointain either of the nodes
    in select_nodes. The output is a dataframe consisting of all the texts that contain the selected nodes or edges.
    """
    network_df= get_network_data(text_df, entity_tags, user_ents, user_dict)

    if operator== "OR":
        mask= network_df['source'].str.lower().isin(list(map(lambda x: x.lower(), select_nodes))) \
        | network_df['target'].str.lower().isin(list(map(lambda x: x.lower(), select_nodes)))
        filtered_df=network_df[mask]
        filtered_df= filtered_df.sort_values('text_id', axis=0)
    elif operator== "AND":
        if len(select_nodes)>2:
            raise ValueError("With the AND operator, you can only enter a list of nodes with two items.")
        mask= network_df['source'].str.lower().isin(list(map(lambda x: x.lower(), select_nodes))) \
        & network_df['target'].str.lower().isin(list(map(lambda x: x.lower(), select_nodes)))
        filtered_df=network_df[mask]
    else:
        raise ValueError("Invalid operator. The operator should be 'AND' or 'OR'.")

    filtered_df= filtered_df.drop_duplicates()
    merged_df = filtered_df.merge(text_df, on='text_id', how='left', suffixes=('_network_df', '_text_df'))

    return merged_df.reset_index(drop=True)
