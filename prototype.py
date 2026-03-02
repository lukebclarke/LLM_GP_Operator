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

    #Convert to numpy to find average
    all_fit_avg_ea_runs = np.array(all_fit_avg_ea)
    all_size_avg_ea_runs = np.array(all_size_avg_ea)
    all_fit_min_ea_runs = np.array(all_fit_min_ea)

    all_fit_avg_ao_runs = np.array(all_fit_avg_ao)
    all_size_avg_ao_runs = np.array(all_size_avg_ao)
    all_fit_min_ao_runs = np.array(all_fit_min_ao)

    #Find mean of runs across each generation
    fit_avg_ea_mean = np.mean(all_fit_avg_ea_runs, axis=0)
    size_avg_ea_mean = np.mean(all_size_avg_ea_runs, axis=0)
    fit_min_ea_mean = np.mean(all_fit_min_ea_runs, axis=0)

    fit_avg_ao_mean = np.mean(all_fit_avg_ao_runs, axis=0)
    size_avg_ao_mean = np.mean(all_size_avg_ao_runs, axis=0)
    fit_min_ao_mean = np.mean(all_fit_min_ao_runs, axis=0)

    #Visualise best solution
    nodes, edges, labels = gp.graph(hof_ao[0])
    plot_tree(nodes, edges, labels)

    #Graphs
    plot_graph("Average Size", size_avg_ea_mean)
    plot_graph("Average Size", size_avg_ao_mean)

    plot_graph("Average Fitness", fit_avg_ea_mean)
    plot_graph("Average Fitness", fit_avg_ao_mean)

    plot_graph("Minimum Fitness", fit_min_ea_mean)
    plot_graph("Minimum Fitness", fit_min_ao_mean)


if __name__ == "__main__":
    main()
