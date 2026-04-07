from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import pickle
import importlib
from daytona import Daytona
import sys
import os
from itertools import combinations

import copydetect

from daytona.common.errors import DaytonaNotFoundError

def pickle_object(obj, file_name):
    """Pickles an object and downloads to /temp folder

    Args:
        obj (Any): Object to pickle
        file_name (string): The name of the file to upload pickled object to
    """
    with open(f"temp/{file_name}.pkl", "wb") as f:
        pickle.dump(obj, f)

def unpickle_object(file_name):
    """Finds object in /temp folder, and unpickles it

    Args:
        file_name (string): The name of the file storing the pickled object

    Returns:
        Any: Unpickled object
    """
    with open(f"temp/{file_name}.pkl", "rb") as f:
        obj = pickle.load(f)

    return obj

def unpickle_daytona_file(file_name, sandbox):
    """Unpickles an object remotely stored, and deletes file afterwards

    Args:
        file_name (string): The name of the file stored in the Daytona sandbox
        sandbox (Daytona): Sandbox

    Returns:
        Any: The unpickled object
    """
    content = sandbox.fs.download_file(f"{file_name}.pkl")

    with open("temp/new_individual.pkl", "wb") as f:
        f.write(content)

    obj = unpickle_object("new_individual")

    #Delete file afterwards
    try:
        sandbox.fs.delete_file(f"{file_name}.pkl")
    except DaytonaNotFoundError:
        #File does not exist
        pass

    return obj

def clean_llm_output(output):
    """Cleans LLM output, extracting the code from it

    Args:
        output (string): The entire LLM output

    Returns:
        string: The pure code from the LLM output
    """
    #Extracts code
    output = output.strip()
    output = output.replace("```", "")
    output = output.replace("python", "")

    #Common bug where mutated_individual keeps getting mistyped
    if "mutuated_individual" in output:
        generated_code = generated_code.replace("mutuated_individual", "mutated_individual")

    return output

def get_similarity(folder_path):
    #Gets all files in directory
    files = os.listdir(folder_path)

    if len(files) <= 1:
        return None
    
    #Finds all combinations of files
    combos = list(combinations(files, 2))

    all_similarities = []
    for pair in combos:
        #Finds filepaths of files
        filepath0 = folder_path + "/" + pair[0]
        filepath1 = folder_path + "/" + pair[1]

        #Compares the similarity between files using the Winnowing algorithm
        fp1 = copydetect.CodeFingerprint(filepath0, 25, 1)
        fp2 = copydetect.CodeFingerprint(filepath1, 25, 1)
        token_overlap, similarities, slices = copydetect.compare_files(fp1, fp2)

        all_similarities.append(similarities[0])
        all_similarities.append(similarities[1])
        
    avg_similarity = sum(all_similarities) / len(all_similarities)

    return avg_similarity