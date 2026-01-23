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

#Visualisation
import pygraphviz as pgv

#LLM
from google import genai
from google.genai import types
from together import Together

#API Keys
from dotenv import load_dotenv

#Files
from evolutionary_algorithm import DynamicOperators

def protectedDiv(left, right):
    # Prevents zero division errors when dividing 
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def plotTree(nodes, edges, labels):
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("tree.pdf")

#Defines Problem
pset = gp.PrimitiveSet("MAIN", 1) #Program takes one input
pset.addPrimitive(operator.add, 2) 
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", partial(random.randint, -1, 1)) #Program can create random constants between 0 and 1
pset.renameArguments(ARG0='x') #Renames input variable to x

def main():
    #Run GP - Simple Evolutionary Algorithm
    algorithm = DynamicOperators(n=300, pset=pset)
    pop, log, hof = algorithm.runSimpleEA()

    print(log)

    #Visualise best solution
    nodes, edges, labels = gp.graph(hof[0])
    plotTree(nodes, edges, labels)

if __name__ == "__main__":
    main()
