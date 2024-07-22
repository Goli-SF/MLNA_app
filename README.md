# Table of Contents

<!-- TOC -->

- [Project Title](#project-title)
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

<!-- /TOC -->

# Project Title
The MultiLingual Network Analysis Package (MLNA)

# Description
This project was funded by the Flex Funds of NFDI4Culture, one of the member consortia of the National Research Data Infrastructure in Germany, in 2024.
It is developed for researchers in the humanities who wish to consult multilingual digital documents for their research, the language of which documents they do not necessarily understand. It allows them to select relevant documents for their research topic from among large digital collections. The intention of developing this package was to integrate digital historical texts in non-European languages into European research and vice versa.

To do so, the package enables you to visulaize network relations between words and entities that you choose to be extracted from the texts. These network visualizations in turn allow you to see which relations between certain keywords (people, places, technologies, dates, products, events, etc.) are mentioned in the texts that you have at your disposal. Having viewed these relations, you can filter out the texts that thematize a certain relation (an edge in the network graph) or contain a certain word (word in the network graph) to be read in future.

# Installation
To install this package from Github, run the following command in the terminal:

```pip install git+https://github.com/Goli-SF/MLNA.git```

You need an internet connection while using MLNA, so that the code can reach the API of Google Translate for translating text from other languages to English. Please do not use a VPN while using this package, so that the connection to Google Translate's API is not interrupted.

I am not sure yet if the package can automatically install the spaCy model that I am using in this package. If this is not the case, please download and install the model using the following code in the terminal:

```python -m spacy download en_core_web_md```

# Usage
I assume that the user of this package has already a number of machine readable texts in different languages at their disposal, including ones the language of which they do not understand. This package is meant for the user to select out the texts that are relevant to their research topic, without having to translate all of them into the languages that they know and reading them all. The user can first filter out those text documents that they need from among their collection, using the MLNA package. Afterwards, they only have to translate the selected texts into a language that they understand and perform a close reading of them.

To do so, you as the user of MLNA can go through the following steps:

1. **Import the modules**

After installing the package, import its modules into your code:

```from mlna import user_input, network```

2. **Prepare text data**

Organize all of the text documents that you have gathered in a pandas dataframe. The mandatory columns in the dataframe are 'text_id' and 'full_text'. The dataframe may contain as many other columns as you desire. I will call this dataframe 'text_df' in the following examples.

3. **Choose entities**

Choose which entities in the texts that you have gathered are relevant to you, using the 'get_entities' function in the 'user_input' module:

```entity_tags= user_input.get_entities()```

This function presents a list of entities that the spaCy package can recognize within the texts. It returns a list of entity tags that is stored in the 'entity_tags' variable in this example. These entity tags help you find keywords in the texts stored in 'text_df'. Using them, you can make sure that you get an overview of the content of each text.

4. **Enter user-defined entities**

If there is a certain word that is not recognized as one of the entities you have selected, but you want it to be included in the text search and network visualization, you can store it in a different variable than 'entites' and use it later in the network visualizations. For example, if you are interested in texts dealing with the development of telegraph and telephone, you can store the word 'telegraph' in a variable like 'user_ents':

```user_ents=['telegraph', 'telephone']```

5. **Create a dictionary**

In order to extract the entities and words selected by the user from multiligual texts, the MLNA package translates all of them into English. It opten happens that proper names in non-English languages are transliterated and therefore appear in different texts in different languages with different spellings. If this is the case, different spellings of the same proper name are recognized by the package as different names. This in turn leads to an invalid representation of network relations between these names and other entities and words selected by the user in the network graph. To correct this, you should run the 'user_dict' function from the 'user_input' module. This function does not only create a dictionary that can be used in your script, it also saves that dictionary as a pickled file onto your computer, in the directory of your code. This way, you can always load the dictionary back into your code and add more key-value pairs to it over time. If you are creating the dictionary from scratch, run:

```
user_dict=user_input.user_dict(text_df, entity_tags, user_ents=None, dict_path=None, threshold=80)
```

If you give the function a dict_path, it will load an existing dictionary and add more key-value-pairs to it. If not, it creates a dictionary from scratch. Either way, the function will prompt you to choose whether you want to see groups of words that could possibly refer to the same proper name. If you choose 'yes', you are shown these word groups and prompted to create a unifies spelling for all of them. The package finds these groups using fuzzy matching. Change the value of 'threshold' in the function to change the accuracy of the fuzzy matching. If you choose 'no', the function prompts you to do this manually by first entering all different spellings that you find to be refering to the same proper name and then entering a unified spelling for that proper name.

Make sure that you run the 'user_dict' function a couple of times on the text dataframe. The reason is this: If you for example have the words 'Naser' and 'Naser al Din Qajar' and want to replace both of them with 'Nasser-al-Din Shah', the function will do the following replacements:

Naser --> Nasser-al-Din Shah

Naser al Din Qajar --> Nasser-al-Din Shah

Now imagine that the package has also detected 'Naser al Din' as an entitiy, but you have forgotten to pass is as one of the keys of the value 'Nasser-al-Din Shah' in the previous step. In this case, the dictionary only replaces the 'Naser' part of this name with 'Nasser-al-Din Shah', therefore turning the original entity into 'Nasser-al-Din Shah al Din' which makes no sense. If you spot such mistakes in the visualized network graph, run the dictionary again and add the key-value pair 'Nasser-al-Din Shah al Din': 'Nasser-al-Din Shah' to it. You can repeat this process as many times as necessary, making sure that you have unified the spellings of all the different variations of the same proper name.

6. **Visualize communities**

If you want to view the communities that exist within the collection of entities from 'entitiy_tags' and words that you have previously stored in 'user_ents', run the 'detect_community' function from the 'network' module:

```
com= network.detect_community (text_df, entity_tags, user_ents=None, user_dict=None, title='community_detection', figsize=(1000, 700), bgcolor='black', font_color='white')
```

This function returns a community visualization stored in your code's path as an .html file. Open this file, interact with the community visualization and see which relationships exist between differnet nodes consisiting of the extracted entities and your selected words.You can change the title of the .html file, its size and the colors of its fonds and background in the function.

Remember, the 'detect_community' function looks at all the texts you have stored in the 'text_df' dataframe.

If you notice wrong or double spellings of a certain word or proper name and want to change them, run the 'user_dict' function again and complement your personalized user dictionary.

7. **Visualize network relations**

If you wish to visualize network relations that exist within the collection of entities from 'entitiy_tags' and words that you have previously stored in 'user_ents', run the 'visualize_network' function:

```
net= netowrk.visualize_network (text_df, entity_tags, user_ents=None, user_dict=None, core=False, select_nodes=None, sources=None, title='network_visualization', figsize=(1000, 700), bgcolor='black', font_color='white')
```

This function returns a network visualization stored in your code's path as an .html file. Open this file, interact with the network visualization and see which relationships exist between differnet nodes consisiting of the extracted entities and your selected words. The relationships (edges) displayed here represent the co-occurence of the nodes in one sentence in one of the texts in 'text_df'. You can change the title of the .html file, its size and the colors of its fonds and background in the function.

If you notice wrong or double spellings of a certain word or proper name and want to change them, run the 'user_dict' function again and complement your personalized user dictionary.

If you only wish to view the core of the network you have just visualized, set `core=True`.If you only wish to visualize a network consisting of a certain group of nodes, store these nodes as a list in `select_nodes`.If you only wish to see the network relations between your predefined entities and selected words in one of the texts from 'text_df', save the 'text_id's of these texts in `sources`.

8. **Select relevant texts**

Having studied the network and community graphs, you have gained a general overview of the subjects and corelations that you could find in the collection of texts that you have gathered. Now is the time to figure out which edges and nodes (which relations and names) appear in which one of the texts in the collection. To do so, run the 'filter_network_data' function from the 'network' module:

```
filtered_df= network.filter_network_data (text_df, select_nodes, entity_tags, user_ents=None, user_dict=None, operator='OR')
```

The argument `select_nodes` stores a list of nodes that you are interested in and want to spot in 'text_df'. If you are interested in one or more nodes, set `operator=OR`. If you are only interested in one edge, namely in the cooccurance of two nodes only, `select_nodes` should contain only these two nodes and you should set `operator=AND`.

The output of this function is a dataframe containing rows from your original dataframe plus three extra rows, containg network data: 'source' and 'target' which include one or both of your selected nodes as well as 'weight' which tells you how many times a certain cooccurance appears in each text.

You can now select the texts displayed in 'filtered_df' and perform a close reading of them, knowing that they very likely contain information that is relevant to your research topic.

# Contributing

Thank you for considering contributing to this project! I very much welcome contributions from the community. Here are the steps to contribute:

**1. Fork the Repository**
   - Click the "Fork" button at the top right of this page to create a copy of this repository in your GitHub account.

**2. Clone the Repository**
   - Clone your forked repository to your local machine using:
     ```git clone https://github.com/Goli-SF/MLNA.git ```

**3. Create a Branch**
   - Create a new branch for your changes:
     ```git checkout -b feature/your-feature-name```

**4. Make Changes**
   - Make your changes or add new features in your branch.

**5. Test Your Changes**
   - Run the tests to ensure your changes do not break anything.

**6. Commit Your Changes**
   - Commit your changes with a clear and concise commit message:
     ```
     git add .
     git commit -m "Add a brief message describing your changes"
     ```

**7. Push to Your Fork**
   - Push your changes to your forked repository:
     ```git push origin feature/your-feature-name```

**8. Open a Pull Request**
   - Go to the original repository on GitHub and click on the "Pull Request" button.
   - Select your branch from the dropdown and click "Create Pull Request".
   - Provide a detailed description of your changes and link any related issues.

### Guidelines
- Ensure your code adheres to the project's coding standards.
- Write clear, concise commit messages.
- Update documentation if necessary.
- Write tests for new features or bug fixes.
- Be respectful and considerate in your communications.

Thank you for your contributions!

# License
This package has a CC BY-NC 4.0 license. For more detail, see the LICENSE.md file. For the third-party licenses of the packages used for the development of MLNA, see the THIRD_PARTY_LICENSES.md file.
