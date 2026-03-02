import operator
import math
import random
import os

import numpy

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
    pop_ao, log_ao, hof_ao = algorithm.runDynamicEA(verbose=True)
    pop_ea, log_ea, hof_ea = algorithm.runSimpleEA()
    algorithm.shutdown_sandbox()

    #Statistics
    fit_avg_ea = log_ea.chapters["fitness"].select("avg")
    size_avgs_ea = log_ea.chapters["size"].select("avg")
    fit_min_ea = log_ea.chapters["fitness"].select("min")
    fit_avg_ao = log_ao.chapters["fitness"].select("avg")
    size_avgs_ao = log_ao.chapters["size"].select("avg")
    fit_min_ao = log_ao.chapters["fitness"].select("min")

    #Visualise best solution
    nodes, edges, labels = gp.graph(hof_ao[0])
    plot_tree(nodes, edges, labels)

    #Graphs
    plot_graph("Average Size", size_avgs_ea)
    plot_graph("Average Size", size_avgs_ao)

    plot_graph("Average Fitness", fit_avg_ea)
    plot_graph("Average Fitness", fit_avg_ao)

    plot_graph("Minimum Fitness", fit_min_ea)
    plot_graph("Minimum Fitness", fit_min_ao)


if __name__ == "__main__":
    main()
