from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import pickle
from daytona import Daytona

def get_individual_from_string(individual_string, pset):
    individual = gp.PrimitiveTree.from_string(individual_string, pset)

    return individual

def get_string_from_individual(individual_obj):
    ind_str = str(individual_obj)

    return ind_str

def pickle_object(obj, file_name):
    with open(f"temp/{file_name}.pkl", "wb") as f:
        pickle.dump(obj, f)

def unpickle_object(file_name):
    with open(f"temp/{file_name}.pkl", "rb") as f:
        obj = pickle.load(f)

    return obj

def unpickle_daytona_file(file_name, sandbox):
    content = sandbox.fs.download_file(f"{file_name}.pkl")

    with open("temp/new_individual.pkl", "wb") as f:
        f.write(content)

    obj = unpickle_object("new_individual")

    return obj

def clean_llm_output(output):
    output = output.strip()
    output = output.replace("```", "")
    output = output.replace("python", "")
    
    return output

def tree_to_list(tree):
    tokens = []
    for node in tree:
        if hasattr(node, "name"):
            tokens.append(node.name)
        else:
            tokens.append(node.value)
    return tokens

def list_to_tree(nodes, pset):
    try:
        tree_str = " ".join(map(str, nodes))
        print(f"Tree_str:\n{tree_str}")
        return gp.PrimitiveTree.from_string(tree_str, pset)
    except:
        raise Exception("Invalid Tree - may be using invalid operators")