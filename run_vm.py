import os
#Setup DEAP
from deap import base, creator, tools, gp
from experiments import tune_gp_model
from experiments import model_comparisons
from experiments import blackbox_vs_groundtruth
from experiments import compare_two_approaches

#Create results folder
try:
    os.mkdir("results")
except FileExistsError:
    pass

#Set up creator object
creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #We want to minimise fitness
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) #Individuals are GP trees (with an associated fitness value)

#TODO: Delete this cell
tuning_ranges = {
    "pop_size": [10], #250
    "cxpb": [0.8],
    "mutpb": [0.1],
    "tourn_size": [3],
    "k": [1,2],
    "self_adapt_req": [None], #Can be set to None (5 works well)
    "default_temperature": [0.3],
    "temperature_alpha": [0.1],
    "model": [None],
    "reasoning_model": [False]
}

tune_gp_model(tuning_ranges, "results/standard_tuning", plot_param=None)
