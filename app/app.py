import streamlit as st
import pandas as pd
import pickle
from googletrans import Translator
from rapidfuzz import fuzz
from langdetect import detect
import pandas as pd
import spacy
import re
import networkx as nx
import pandas as pd
import community.community_louvain as community_louvain
from pyvis.network import Network



# user_input:
def get_entities():
    """
    Prompts the user to enter the entities they are looking for in the input texts. The entities stem form spaCy.
    """
    options= ['People, including fictional.',
              'Nationalities or religious or political groups.',
              'Buildings, airports, highways, bridges, etc.',
              'Companies, agencies, institutions, etc.',
              'Countries, cities, states.',
              'Non-GPE locations, mountain ranges, bodies of water.',
              'Objects, vehicles, foods, etc. (Not services.)',
              'Named hurricanes, battles, wars, sports events, etc.',
              'Titles of books, songs, etc.',
              'Named documents made into laws.',
              'Any named language.',
              'Absolute or relative dates or periods.',
              'Monetary values, including unit.']

    options_dict= {'People, including fictional.': 'PERSON',
              'Nationalities or religious or political groups.': 'NORP',
              'Buildings, airports, highways, bridges, etc.': 'FAC',
              'Companies, agencies, institutions, etc.': 'ORG',
              'Countries, cities, states.': 'GPE',
              'Non-GPE locations, mountain ranges, bodies of water.': 'LOC',
              'Objects, vehicles, foods, etc. (Not services.)': 'PRODUCT',
              'Named hurricanes, battles, wars, sports events, etc.': 'EVENT',
              'Titles of books, songs, etc.': 'WORK_OF_ART',
              'Named documents made into laws.': 'LAW',
              'Any named language.': 'LANGUAGE',
              'Absolute or relative dates or periods.': 'DATE',
              'Monetary values, including unit.': 'MONEY'}


    entities = st.multiselect("**Which entities are you looking for in the texts?**", options)
    entity_tags= []
    for entity in entities:
        entity_tags.append(options_dict[entity])
    return entity_tags


def get_user_ents():
    """
    Prompts the user to enter the words they are looking for in the input texts, other than the entities.
    """
    all_ents= st.text_input("**Enter as many words as you wish to include in the network analysis, that do not fit under the entity categories. Separate the words with a comma. Press Enter when finished.**")
    user_ents= all_ents.split(',')
    return user_ents


def select_nodes (text_df, entity_tags, user_ents=None, user_dict=None):
    """
    Prompts the user to enter the names of the nodes they want to extract from the network data. These nodes can be
    used to visualize a network consisting only of them or to filter relevant texts from the text_df based on the
    nodes/entities present in the texts. The node names entered by the user should match the entities recognized by
    the extract_entities function from the preproc module.
    """
    network_df= get_network_data(text_df, entity_tags, user_ents, user_dict)
    all_nodes= st.text_input ("**Enter the nodes that you want to see in the network graph or in the filtered DataFrame. Separate the words with a comma. Press Enter when finished.**")
    selected_nodes= all_nodes.split(',')
    for node in selected_nodes:
        if node.lower() not in list(map(lambda x: x.lower(), network_df['source'])) and\
        node.lower() not in list(map(lambda x: x.lower(), network_df['target'])):
            selected_nodes.remove(node)

    return selected_nodes


def user_dict (text_df, entity_tags, user_ents=None, threshold=80):
    """
    This function allows users to set a preferred spelling for proper names and convert all variations to this standard
    version. The dictionary is saved to the code's path, enabling it to be reloaded and updated at different stages of
    using the package. To reload and further develop the dictionary, users need to enter the path to the existing
    pickled dictionary into the dict_path argument. Additionally, users can adjust the threshold value to fine-tune
    the fuzzy matching.
    """
    dict_answer= st.radio ("**Do you already have a dictionary saved locally on your computer?**", ["Yes", "No"])

    if dict_answer== "Yes":
        dict_path= st.file_uploader("**Enter the path to the locally saved dictionary:**", type="pickle")
        user_dict = pickle.load(dict_path)
        st.write("**Here's the uploaded dictionary:**")
        dict_items= list(user_dict.items())
        dict_df= pd.DataFrame(dict_items, columns=['key', 'value'])
        st.dataframe(dict_df)

        expand_answer= st.radio ("**Do you wish to expand this dictionary?**", ["Yes", "No"])
        if expand_answer== "Yes":
            expand_answer= st.radio ("How would you like to expand the dictionary?", ["Expand manually", "View suggestions"])
            if expand_answer== "Expand manually":
                ####








    if dict_answer== "No":
        pass





    else:
        user_dict={}

    fuzz_prompt= input("Do you wish to see all similar entities and define a constant spelling for them? Enter 'y' for Yes and 'n' for No: ")
    print()

    if fuzz_prompt.lower()=='n':
        constant= input("Enter the standard spelling of an entity: ")
        print()
        print(f"Enter all vatiations of '{constant}' that exist among the entities. Enter 'done' to exit. Enter 'next' to set another standard spelling for another entity.")
        print()
        while True:
            variation= input(f"Enter a varying spelling of '{constant}': ")
            if variation.lower()== 'done':
                break
            elif variation.lower()== 'next':
                print()
                constant= input("Enter the standard spelling of an entity: ")
                print()
                print(f"Enter all vatiations of '{constant}' that exist among the entities. Enter 'done' to exit. Enter 'next' to set another standard spelling for another entity.")
                print()
            else:
                user_dict[variation]= constant

    elif fuzz_prompt.lower()=='y':
        similar_groups= group_similar_ents (text_df, entity_tags, user_ents, user_dict, threshold)
        for group in similar_groups:
            print()
            print("The following words seem to refer to the same entity:")
            print()
            for ent in group:
                print (ent)
            print()
            unified_entity= input("Enter a standard spelling for all of the above entities. Enter 's' to skip to the next set. Enter 'done' to exit.")
            if unified_entity.lower()=='done':
                break
            elif unified_entity.lower()!='s':
                for ent in group:
                    user_dict[ent]= unified_entity

    if dict_path:
        with open(dict_path, 'wb') as f:
            pickle.dump(user_dict, f)
    else:
        file_name= 'user_dict.pickle'
        with open(file_name, 'wb') as f:
            pickle.dump(user_dict, f)

    return user_dict



# preproc:
nlp = spacy.load("en_core_web_md")

def find_sign_index (text):
    """
    Gets a string and returns the index of the first occurance of the signs that mark a sentence [. ! ?] in the string.
    """
    match = re.search(r"[.!?]", text)
    if match:
        return match.start()
    else:
        return None


def apply_user_dict(text, user_dict):
    """
    Replaces the keys of a user-defined dictionary with their values in a long text.
    """
    for key, value in user_dict.items():
        if key in text:
            text= text.replace(key, value)
    return text


def translate_long_text (text):
    """
    Translates long texts into English. This function was developed because the Translator model does not translate
    texts that are longer than 5000 characters. So this function devides the input text into chunks of 4000
    characters, translates them and puts them together again.
    """
    source= detect(text)
    translator = Translator()
    translation= ''

    while True:
        if len(text) <= 4000:
            translation += translator.translate(text, src=source, dest="en").text
            break
        else:
            fun_text= text[:4000] + "<placeholder>" + text[4000:]
            ph_index= fun_text.index("<placeholder>")
            rest= text[ph_index:]
            sign_index= find_sign_index(rest)
            cut_index= ph_index + sign_index
            chunk= text[:cut_index]
            translation += translator.translate(chunk, src=source, dest="en").text + ". "
            text = text.replace(chunk, "", 1)

    translation= translation.replace("..", ".")

    return translation


def sent_tokenize (eng_text, user_dict=None):
    """
    Sentence-tokenizes a text that is in English. I did not use NLTK's sentence tokenizer, becauase it cannot recognize
    sentences when there is no space between the full stop at the end of the sentence and the next word.
    """
    if user_dict:
        eng_text= apply_user_dict(eng_text, user_dict)
        eng_text= apply_user_dict(eng_text, user_dict)

    eng_sentences= [sentence.strip() + '.' for sentence in re.split(r'[.!?]', eng_text) if sentence.strip()]

    return eng_sentences


def trans_sent_tokenize (text, user_dict=None):
    """
    Translates the input text to English if it is not already in English. Then sentence-tokenizes the resulting
    English text. It returns a tuple of the original language of the text and a list of the tokenized English sentences.
    """
    language= detect(text)

    if language != 'eng':
        eng_text= translate_long_text(text)
        eng_sentences= sent_tokenize(eng_text, user_dict)
    else:
        eng_sentences= sent_tokenize(text, user_dict)

    return eng_sentences


def extract_entities (text, text_id, entity_tags, user_ents=None, user_dict=None):
    """
    This function receives a whole text, translates it to English (if it is not already in English) and
    sentence-tokenizes it. It then extracts the desired entities that are given as a list and stored in
    entity_tags. It also extracts all desired words by the user, stored in user_ents, from the text. It finally stores
    all of the entities and words in a dictionary with the following keys: 'text_id', 'sentences', 'entities'.
    """
    eng_sentences = trans_sent_tokenize(text, user_dict)
    entities = []

    for sent in eng_sentences:
        sent_entities=[]
        sent_doc = nlp(sent)
        for ent in sent_doc.ents:
            if ent.label_ in entity_tags:
                # doing some text cleaning:
                entity = ent.text.strip()
                if "'s" in entity:
                    cutoff = entity.index("'s")
                    entity = entity[:cutoff]
                if "’s" in entity:
                    cutoff = entity.index("’s")
                    entity = entity[:cutoff]
                if "ʿ" in entity:
                    entity.replace("ʿ", "")
                if entity != "":
                    sent_entities.append(entity)
        seen = set()
        sent_entities= [x for x in sent_entities if not (x in seen or seen.add(x))]

        # adding the words defined by the user as fixed entities to the entity list:
        if user_ents:
            funny_sent= sent
            for ent in sent_entities:
                funny_sent.replace(ent, str(sent_entities.index(ent)))
            funny_sent_words= [x.lower() for x in sent.split()]
            source_index= 0
            for i in range(len(funny_sent_words)):
                if funny_sent_words[i]=='1':
                    source_index=i
                    break
            for word in user_ents:
                word=word.lower()
                if word in funny_sent_words:
                    # word_index = index of each single word provided by the user within the original sentence:
                    word_index=funny_sent_words.index(word)
                    if word_index <= source_index:
                        sent_entities.insert(0, word)
                    else:
                        sent_entities.append(word)

        entities.append(sent_entities)

    ent_dict= {'text_id': text_id,
               'sentences': eng_sentences,
               'entities': entities
                }

    return ent_dict


def group_similar_ents(text_df, entity_tags, user_ents=None, user_dict=None, threshold=80):
    """
    Finds similar entities using fuzzy matching. It will be used in the user_dict function in the user_input module to
    suggest similar dictionary keys to the user.
    """
    # helper function that creates groups of similar entities:
    def find_group(groups, ent):
        for group in groups:
            if any(fuzz.ratio(ent, member) >= threshold for member in group):
                return group
        return None

    groups = []

    for i, row in text_df.iterrows():
        text= row['full_text']
        text_id= row['text_id']

        ent_lists= extract_entities (text, text_id, entity_tags, user_ents, user_dict)['entities']

        for ent_list in ent_lists:
            if len(ent_list)>0:
                for ent in ent_list:
                    group = find_group(groups, ent)
                    if group:
                        if ent not in group:
                            group.append(ent)
                    else:
                        groups.append([ent])

    groups = [group for group in groups if len(group) >= 2]

    return groups



# network:
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
    figsize=(700, 500), bgcolor='black', font_color='white'):
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
    title='network_visualization', figsize=(700, 500), bgcolor='black', font_color='white'):
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



# app:
def main():

    st.title ('MLNA')
    st.write ('### The MultiLingual Network Analysis package')
    st.link_button ('Visit the MLNA repo on GitHub.', url='https://github.com/Goli-SF/MLNA')

    # Prompt the user to upload a file:
    uploaded_file= st.file_uploader("**Upload a pickled DataFrame containing your texts (up to 200 MB):**", type="pickle")
    if uploaded_file is not None:
        # Read the file into a DataFrame:
        text_df= pd.read_pickle(uploaded_file)
        # Display the DataFrame:
        st.write("**Here's the uploaded DataFrame:**")
        st.dataframe(text_df)


    st.write('#### User Dictionary')
    st.markdown("""---""")

    ####




    col1, col2= st.columns(2)
    with col1:
        st.write('#### Predefined Entities')
        st.markdown("""---""")
        # Prompt the user to enter the entities they are looking for:
        entity_tags = get_entities()
        if len(entity_tags)>0:
            st.write ("**Here's your entitiy list:**", entity_tags)

    with col2:
        st.write('#### User-Defined Entities')
        st.markdown("""---""")
        # Prompt the user to enter self-defined entities:
        user_ents= get_user_ents()
        if len(user_ents)>0:
            st.write ("**Here's your list of words:**", user_ents)

    # with col3:
    #     st.write('#### list of selected nodes')
    #     st.markdown("""---""")
    #     # Prompt the user to enter self-defined nodes:
    #     selected_nodes= select_nodes (text_df, entity_tags=entity_tags, user_ents=user_ents, user_dict=None)
    #     if len(selected_nodes)>0:
    #         st.write ("**Here's your list of selected nodes:**", selected_nodes)

    # st.write('#### network graph')
    # st.markdown("""---""")
    # html_content = visualize_network (text_df, entity_tags=entity_tags, user_ents=user_ents, user_dict=None, core=True, select_nodes=None, sources=None,\
    # title='network_visualization', figsize=(700, 500), bgcolor='black', font_color='white')
    # st.components.v1.html(html_content, width=700, height=500)

        # content= detect_community (text_df, ['PERSON', 'GPE'], user_ents=None, user_dict=None, title='community_detection',\
        # figsize=(800, 600), bgcolor='black', font_color='white')
        # st.components.v1.html(content, height=500, width=700)

if __name__ == "__main__":
    main()
