#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

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

### Graphviz Section ###
def plotTree(nodes, edges, labels):
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("tree.pdf")

def createMutationPrompt(base_prompt, individual):
    #Creates a custom mutation prompt that includes information about the individual
    #TODO: Include fitness, avg fitness, etc.
    #TODO: More optimal way to count number of nodes?
    nodes, edges, labels = gp.graph(individual)
    num_nodes = len(nodes)
    height = individual.height
    additional_info = f"The current GP individual:\n- Has {num_nodes} nodes\n- Has a depth of {height}\n"

    return additional_info + base_prompt

def setupLLM_Gemini(mutation_prompt_file):
    #Defines LLM Client for custom genetic operators
    api_key = os.environ.get("GEMINI_KEY")

    if api_key is None:
        raise Exception("API key not found")

    client = genai.Client(api_key=api_key)

    #Gets prompt for custom mutation
    f = open(mutation_prompt_file)
    base_mutation_prompt = f.read()

    #TODO: Gets prompt for custom crossover

    return client, base_mutation_prompt

def setupLLM_Together(mutation_prompt_file):
    #Defines LLM Client for custom genetic operators
    api_key = os.environ.get("TOGETHER_AI")

    if api_key is None:
        raise Exception("API key not found")

    client = Together(api_key=api_key)
    
    #Gets prompt for custom mutation
    f = open(mutation_prompt_file)
    base_mutation_prompt = f.read()

    #TODO: Gets prompt for custom crossover

    return client, base_mutation_prompt

def getLLMChoice(individual, client, base_prompt):
    #Returns a number corresponding to a choice of strategy
    full_prompt = createMutationPrompt(base_prompt, individual)

    while True:
        try:
            response = client.chat.completions.create(
                model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                temperature=0.95,
                messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
                ]
            )
            
            print(response.choices[0].message.content)
            choice = int(response.choices[0].message.content)

            #TODO: Ensure LLM choice is valid (within options)

            return choice
        #Ensures that the choice is an integer
        except ValueError:
            print("Retry")
            pass

def selectMutation_LLM(individual, client, base_prompt):
    #TODO: Provide more information to the LLM so it can better select an individual
    choice = getLLMChoice(individual, client, base_prompt)

    if choice == 1:
        #Uniform Mutation
        print("Uniform Mutation")
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)
    elif choice == 2:
        #Gaussian Mutation
        print("Shrink Mutation")
        return gp.mutShrink(individual)
    elif choice == 3:
        print("Node replacement")
        return gp.mutNodeReplacement(individual, pset=pset)
    else:
        #TODO: Handle this error 
        print("Invalid choice")
        raise ValueError

# Custom Mutation
def customMutate(individual, client, base_prompt):
    #TODO: Identify a point at which evolution stagnates
    #For now, we will use custom mutate randomly
    createMutationPrompt(base_prompt, individual)
    if random.random() < 0.05:
        #Custom LLM Mutation
        return selectMutation_LLM(individual, client, base_prompt)
    else:
        #Otherwise, just perform mutation as normal
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)

# Define new functions
def protectedDiv(left, right):
    # Prevents zero division errors when dividing 
    try:
        return left / right
    except ZeroDivisionError:
        return 1

#Loads in environment variables
load_dotenv()

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

creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #We want to minimise fitness
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) #Individuals are GP trees (with an associated fitness value)

# Defines 'toolbox' functions we can use to create and evaluate individuals
toolbox = base.Toolbox() 
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2) #Generates random expressions (some full trees, other small ones)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr) #Creates individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual) #Creates populations
toolbox.register("compile", gp.compile, pset=pset) #Converts tree into runnable code 

def evalSymbReg(individual, points):
    #Defines the fitness function for an individual 
    func = toolbox.compile(expr=individual) #Transform the tree expression in a callable function

    # Evaluate the mean squared error between the expression and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    return math.fsum(sqerrors) / len(points),

# Defines genetic operators
client, mutation_prompt = setupLLM_Together("LLMPromptMutation.txt") #Custom operators require LLM input
toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10,10)]) #Training data is between -1 and 1
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", customMutate, client=client, base_prompt=mutation_prompt)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)) #Limits height of tree
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1) #We track 1 best solution

    # Track statistics
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    # Run GP - Simple Evolutionary Algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log    

    #Visualise best solution
    nodes, edges, labels = gp.graph(hof[0])
    plotTree(nodes, edges, labels)

    return pop, log, hof

if __name__ == "__main__":
    main()
