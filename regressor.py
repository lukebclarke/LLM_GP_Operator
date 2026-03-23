from adaptive_regressor import AdaptiveRegressor

from deap import gp, base, creator
from sympy import sympify
import inspect
import networkx as nx
import pygraphviz as pgv
import numpy as np

#Testing - TODO: Can remove
import operator
import gp_primitives
import math
from functools import partial
import random
import inspect


#TODO: Hyperparameter tuning by wrapping in sklearn CV class

est = AdaptiveRegressor(
    pop_size=200,
    gens=40,
    max_time=8.0*60.0*60.0,
    cxpb=0.7,
    mutpb=0.1,
    k=3,
    verbose=True
)

def get_string(tree, index=None, variable_mapping={}):
    """  
    Converts a DEAP expression to a sympy compatible model

    Parameters
    ----------
    tree: gp.Individual
        The DEAP GP individual
    
    index: int
        The position of the node to inspect within the tree

    variable_mapping: dict
        Provides a mapping between DEAP variable names (e.g. ARG0) and the names provided by the data (e.g. x) 

    Returns
    -------
    A sympy-compatible string of the final model. 
    """
    
    #Defines how to handle each term
    bracket_terms = ["log", "protectedDiv", "protectedRoot", "sin", "cos", "tan", "exp"]
    sympy_mapping = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "protectedDiv": "/",
        "protectedRoot": "sqrt",
        "cube": "**3",
        "square": "**2"
    }

    if index == None:
        current_node = tree.root
    else:
        current_node = tree[index]

    #Converts to string by firstly converting to PrimitiveTree, and then using DEAP's built in string conversion
    current_node = str(current_node.name)

    #TODO: Better way to do this?
    #Converts any names to sympy formats
    for k,v in sympy_mapping.items():
        current_node = current_node.replace(k,v)

    #Replaces with Sympy compatible terms
    current_node = sympy_mapping.get(current_node, current_node)
    current_node = variable_mapping.get(current_node, current_node)

    children = get_children_indices(tree, index)
    node_str = ""
    #If no children, return the string as is 
    if not children:
        node_str = current_node
    #Arity 1
    elif len(children) == 1:
        #Put the term in brackets
        if current_node in bracket_terms:
            node_str = current_node + "(" + get_string(tree, children[0], variable_mapping) + ")"
        #Put operator (e.g. **2) after the term
        else:
            node_str = "(" + get_string(tree, children[0], variable_mapping) + current_node + ")"
    #Arity 2
    elif len(children) == 2:
        left_child = get_string(tree, children[0], variable_mapping)
        right_child = get_string(tree, children[1], variable_mapping)

        node_str = "(" + left_child + current_node + right_child + ")"
    else:
        raise Exception("Invalid arity")

    return node_str
            
def get_children_indices(tree, index):
    node = tree[index]

    #Terminal nodes have no children (i.e. empty list)
    if node.arity == 0:
        return [] 
    
    children_indices = []
    #First child
    i = index + 1 
    for _ in range(node.arity):
        #Finds out number of nodes in child's subtree
        children_indices.append(i)
        i += len(tree[tree.searchSubtree(i)])
    
    return children_indices

def model(est, X=None):
    """
    Return a sympy-compatible string of the final model. 

    Parameters
    ----------
    est: sklearn regressor
        The fitted model. 
    X: pd.DataFrame, default=None
        The training data. This argument can be dropped if desired.

    Returns
    -------
    A sympy-compatible string of the final model. 
    """

    #Finds best model
    new_model = est.hof_[0]

    #Maps variable names in model to variable names in training data
    mapping = {'ARG'+str(i):k for i,k in enumerate(X.columns)}

    model_str = get_string(new_model, variable_mapping=mapping)

    return model_str

def get_testing_data():
    X = np.array([])
    Y = np.array([])
    for i in range(500):
        x = random.uniform(0, 1000)
        y = (x**3) + (x**2) + (x) + 1
        X = np.append(X, x)
        Y = np.append(Y, y)

    print(X)
    
    X.reshape(-1, 1)
    Y.reshape(-1, 1)

    return X, Y

def test_model():
    X, Y = get_testing_data()
    est.fit(X, Y)
    print(model(est))

test_model()