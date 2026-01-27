from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


def get_individual_from_string(individual_string, pset):
    individual = gp.PrimitiveTree.from_string(individual_string, pset)

    return individual

def get_string_from_individual(individual_obj):
    ind_str = str(individual_obj)

    return ind_str