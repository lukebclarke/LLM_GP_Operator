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
from custom_mutate import CustomMutate

class DynamicOperators():
    def __init__(self, n, pset):
        random.seed(318)
        self.n = n
        self.pset = pset

        #Loads in environment variables
        load_dotenv()

        #TODO: Maybe move this somewhere else - we pass the creator? 
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #We want to minimise fitness
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) #Individuals are GP trees (with an associated fitness value)

        # Defines 'toolbox' functions we can use to create and evaluate individuals
        self.toolbox = base.Toolbox() 
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2) #Generates random expressions (some full trees, other small ones)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr) #Creates individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual) #Creates populations
        self.toolbox.register("compile", gp.compile, pset=pset) #Converts tree into runnable code 
        
        # Defines genetic operators
        client, mutation_prompt = self.setupLLM("LLMPromptMutation.txt") #Custom operators require LLM input
        self.toolbox.register("evaluate", self.evaluateIndividual, points=[x/10. for x in range(-10,10)]) #Training data is between -1 and 1
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)

        self.mutator = CustomMutate(client, mutation_prompt, self.pset, self.toolbox)
        self.toolbox.register("mutate", self.mutator.mutate, client=client, base_prompt=mutation_prompt) #Uses custom mutation function
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)) #Limits height of tree

        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        #Initialises population
        self.pop = self.toolbox.population(n=300)
        self.hof = tools.HallOfFame(1) #We track 1 best solution

        # Track statistics
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        self.mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        self.mstats.register("avg", numpy.mean)
        self.mstats.register("std", numpy.std)
        self.mstats.register("min", numpy.min)
        self.mstats.register("max", numpy.max)

    def setupLLM(self, mutation_prompt_file):
        #Defines LLM Client for custom genetic operators
        api_key = os.environ.get("TOGETHER_AI") #Uses the TogetherAI API

        if api_key is None:
            raise Exception("API key not found")

        client = Together(api_key=api_key)
        
        #Gets prompt for custom mutation
        f = open(mutation_prompt_file)
        base_mutation_prompt = f.read()

        #TODO: Gets prompt for custom crossover

        return client, base_mutation_prompt
    
    def evaluateIndividual(self, individual, points):
        #TODO: This is temporary for this specific function
        #Defines the fitness function for an individual 
        func = self.toolbox.compile(expr=individual) #Transform the tree expression in a callable function

        # Evaluate the mean squared error between the expression and the real function : x**4 + x**3 + x**2 + x
        sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
        return math.fsum(sqerrors) / len(points),

    def runSimpleEA(self):
        # Run GP - Simple Evolutionary Algorithm
        pop, log = algorithms.eaSimple(self.pop, self.toolbox, 0.5, 0.1, 40, stats=self.mstats,
                                    halloffame=self.hof, verbose=True)
        
        return pop, log, self.hof