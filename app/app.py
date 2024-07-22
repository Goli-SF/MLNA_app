import streamlit as st
import pandas as pd
from mlna import network

# import sys
# import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# project_dir = os.path.join(parent_dir, 'mlna')
# sys.path.append(project_dir)
# import user_input, preproc, network



def main():

    st.title ('MLNA')
    st.write ('### The MultiLingual Network Analysis package')
    st.link_button ('Visit the MLNA repo on Github.', url='https://github.com/Goli-SF/MLNA')

    # Prompt the user to upload a file
    uploaded_file= st.file_uploader("Upload a pickled data frame containing your texts:", type="pickle")
    if uploaded_file is not None:
        # Read the file into a DataFrame
        text_df= pd.read_pickle(uploaded_file)
        # Display the DataFrame
        st.write("Uploaded data frame:")
        st.dataframe(text_df)

        # html_content = network.visualize_network (text_df, ['PERSON'], user_ents=None, user_dict=None, core=True, select_nodes=None, sources=None,\
        # title='network_visualization', figsize=(1000, 700), bgcolor='black', font_color='white')
        # st.components.v1.html(html_content, height=500, width=700)

        content= network.detect_community (text_df, ['PERSON', 'GPE'], user_ents=None, user_dict=None, title='community_detection',\
        figsize=(800, 600), bgcolor='black', font_color='white')
        st.components.v1.html(content, height=500, width=700)

if __name__ == "__main__":
    main()
