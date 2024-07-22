import pickle
from network import get_network_data
from preproc import group_similar_ents



def get_entities ():
    """
    Prompts the user to enter the entities they are looking for in the input texts. The entities stem form spaCy.
    """
    options= {'PERSON': 'People, including fictional.',
              'NORP': 'Nationalities or religious or political groups.',
              'FAC':'Buildings, airports, highways, bridges, etc.',
              'ORG': 'Companies, agencies, institutions, etc.',
              'GPE': 'Countries, cities, states.',
              'LOC': 'Non-GPE locations, mountain ranges, bodies of water.',
              'PRODUCT': 'Objects, vehicles, foods, etc. (Not services.)',
              'EVENT': 'Named hurricanes, battles, wars, sports events, etc.',
              'WORK_OF_ART': 'Titles of books, songs, etc.',
              'LAW': 'Named documents made into laws.',
              'LANGUAGE': 'Any named language.',
              'DATE': 'Absolute or relative dates or periods.',
              'MONEY': 'Monetary values, including unit.'}
    entity_tags = []
    print("Enter the entities you are looking for. Enter 'done' when finished.")
    print()
    for key in options.keys():
        print (f"{key} : {options[key]}")
    print()
    while True:
        choice = input("Enter your choice: ").upper()
        if choice == 'DONE':
            break
        elif choice in options:
            entity_tags.append(choice)
        else:
            print("Invalid option. Please select a valid option or enter 'done' to exit.")

    return entity_tags



def select_nodes (text_df, entity_tags, user_dict=None, user_ents=None):
    """
    Prompts the user to enter the names of the nodes they want to extract from the network data. These nodes can be
    used to visualize a network consisting only of them or to filter relevant texts from the text_df based on the
    nodes/entities present in the texts. The node names entered by the user should match the entities recognized by
    the extract_entities function from the preproc module.
    """
    network_df= get_network_data(text_df, entity_tags, user_dict, user_ents)
    select_nodes=[]
    print ("Enter the names of as many nodes as you wish. Enter 'done' to exit.")
    print()
    while True:
        node= input("Enter the name of a node: ")
        if node.lower()=='done':
            break
        elif node.lower() not in list(map(lambda x: x.lower(), network_df['source'])) or\
        node.lower() not in list(map(lambda x: x.lower(), network_df['target'])):
            print()
            print ("Invalid input. Please enter a valid node or enter 'done' to exti.")
        else:
            select_nodes.append(node)

    return select_nodes



def user_dict (text_df, entity_tags, user_ents=None, dict_path=None, threshold=80):
    """
    This function allows users to set a preferred spelling for proper names and convert all variations to this standard
    version. The dictionary is saved to the code's path, enabling it to be reloaded and updated at different stages of
    using the package. To reload and further develop the dictionary, users need to enter the path to the existing
    pickled dictionary into the dict_path argument. Additionally, users can adjust the threshold value to fine-tune
    the fuzzy matching.
    """
    if dict_path:
        with open(dict_path, 'rb') as f:
            user_dict = pickle.load(f)
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
        similar_groups= group_similar_ents (text_df, entity_tags, user_dict, user_ents, threshold)
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
