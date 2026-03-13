import operator
import math
import random
import os

import numpy as np

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import gp_primitives

#Visualisation
import pygraphviz as pgv
import matplotlib.pyplot as plt

#LLM
from google import genai
from google.genai import types
from together import Together

#API Keys
from dotenv import load_dotenv

#Files
from evolutionary_algorithm import DynamicOperators

def plot_tree(nodes, edges, labels):
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("docs/tree.pdf")

def plot_graph(metric_name, metric_values):
    print(metric_values)
    xdata = list(range(0, len(metric_values), 1))
    ydata = metric_values

    fig = plt.figure(figsize=[7,5])
    ax = plt.subplot(111)
    ax.plot(xdata, ydata)  # <-- Plot the data
    ax.set_xlabel("Generations")
    ax.set_ylabel(metric_name)
    ax.grid('on')

    plt.show()

def get_stats(all_fit_avg, all_size_avg, all_fit_min):
    #Ensure all runs are of same length (e.g. padding)
    max_length_alg = max((len(run)) for run in all_fit_avg)

    padded_avg_fitnesses = []
    padded_avg_size = []
    padded_min_fit = []

    #Finds padded metrics for each run
    for i in range(len(all_fit_avg)):
        padded_avg_fitnesses.append(all_fit_avg[i] + ([0] * (max_length_alg - len(all_fit_avg[i]))))
        padded_avg_size.append(all_size_avg[i] + ([0] * (max_length_alg - len(all_size_avg[i]))))
        padded_min_fit.append(all_fit_min[i] + ([0] * (max_length_alg - len(all_fit_min[i]))))

    #Convert to numpy to find average
    all_fit_avg_runs = np.array(padded_avg_fitnesses)
    all_size_avg_runs = np.array(padded_avg_size)
    all_fit_min_runs = np.array(padded_min_fit)

    #Find mean of runs across each generation
    fit_avg_mean = np.mean(all_fit_avg_runs, axis=0)
    size_avg_mean = np.mean(all_size_avg_runs, axis=0)
    fit_min_mean = np.mean(all_fit_min_runs, axis=0)

    return fit_avg_mean, size_avg_mean, fit_min_mean


def plot_comparison_graph(metric_name, alg1_label, alg2_label, metric_values1, metric_values2):
    xdata1 = list(range(0, len(metric_values1), 1))
    ydata1 = metric_values1

    xdata2 = list(range(0, len(metric_values2), 1))
    ydata2 = metric_values2

    fig = plt.figure(figsize=[7,5])
    ax = plt.subplot(111)
    ax.plot(xdata1, ydata1, label=alg1_label) 
    ax.plot(xdata2, ydata2, label=alg2_label) 
    ax.set_xlabel("Generations")
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.grid('on')

    plt.show()

#Defines Problem
pset = gp.PrimitiveSet("MAIN", 1) #Program takes one input
pset.addPrimitive(operator.add, 2) 
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(gp_primitives.protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", partial(random.randint, -1, 1)) #Program can create random constants between 0 and 1
pset.renameArguments(ARG0='x') #Renames input variable to x

def main():
    #Run GP - Simple Evolutionary Algorithm
    algorithm = DynamicOperators(n=300, pset=pset, k=5)
    num_runs = 10

    #Statistics
    all_fit_avg_ea = []
    all_size_avg_ea = []
    all_fit_min_ea = []

    all_fit_avg_ao = []
    all_size_avg_ao = []
    all_fit_min_ao = []

    for i in range(num_runs):
        print("Run number: ", i)
        pop_ao, log_ao, hof_ao = algorithm.runDynamicEA(verbose=True)
        pop_ea, log_ea, hof_ea = algorithm.runSimpleEA()

        all_fit_avg_ea.append(log_ea.chapters["fitness"].select("avg"))
        all_size_avg_ea.append(log_ea.chapters["size"].select("avg"))
        all_fit_min_ea.append(log_ea.chapters["fitness"].select("min"))

        all_fit_avg_ao.append(log_ao.chapters["fitness"].select("avg"))
        all_size_avg_ao.append(log_ao.chapters["size"].select("avg"))
        all_fit_min_ao.append(log_ao.chapters["fitness"].select("min"))

    algorithm.shutdown_sandbox()

    #Gets statistics across all runs
    ea_fit_avg, ea_size_avg, ea_fit_min = get_stats(all_fit_avg_ea, all_size_avg_ea, all_fit_min_ea)
    ao_fit_avg, ao_size_avg, ao_fit_min = get_stats(all_fit_avg_ao, all_size_avg_ao, all_fit_min_ao)

    #Visualise best solution
    nodes, edges, labels = gp.graph(hof_ao[0])
    plot_tree(nodes, edges, labels)

    #Graphs
    plot_comparison_graph("Average Size", "Standard", "Adaptive Operator", ea_size_avg, ao_size_avg)
    plot_comparison_graph("Average Fitness", "Standard", "Adaptive Operator", ea_fit_avg, ao_fit_avg)
    plot_comparison_graph("Minimum Fitness", "Standard", "Adaptive Operator", ea_fit_min, ao_fit_min)

if __name__ == "__main__":
    main()
