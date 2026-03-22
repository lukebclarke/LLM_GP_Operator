import operator
import math
import random
import os
import numpy as np

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
from daytona import Daytona
import subprocess



#API Keys
from dotenv import load_dotenv

#Files
from custom_mutate import CustomMutate
from custom_crossover import CustomCrossover


#TODO: Define max num retires in here
class DynamicOperators():
    def __init__(self, n, pset, k=2):
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
        self.client = self.setupLLM() #Custom operators require LLM input
        self.sandbox = self.setupDaytona()

        self.toolbox.register("evaluate", self.evaluateIndividual, points=[x/10. for x in range(-10,10)]) #Training data is between -1 and 1
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)

        #Defines custom mutation + crossover interfaces
        self.mutator = CustomMutate(self.client, self.sandbox, self.pset, self.toolbox, creator, model="Qwen/Qwen3-Coder-Next-FP8", max_num_retries=10, max_local_skips=5)
        self.custom_crossover = CustomCrossover(self.client, self.sandbox, self.pset, self.toolbox, creator, model="Qwen/Qwen3-Coder-Next-FP8", max_num_retries=10, max_local_skips=3)

        #Registers custom mutation + crossover methods
        self.toolbox.register("mate", self.custom_crossover.crossover)
        self.toolbox.register("mutate", self.mutator.mutate) 

        #Defines limits for genetic operations
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
        self.fitness_improvements = []
        self.redesign_generations = []

        self.k = k #Redesign algorithm if there has no improvement in fitness for K generations
        self.gens_since_improvement = 0
        self.prev_avg_fitness = np.inf

    def reset_state(self):
        #Used for when algorithms are run multiple times
        self.pop = self.toolbox.population(n=self.n)
        self.hof = tools.HallOfFame(1) #We track 1 best solution
        self.fitness_improvements = []
        self.gens_since_improvement = 0
        self.prev_avg_fitness = 100000
        self.mutator.reset_operator()
        self.custom_crossover.reset_operator()

    def setupLLM(self):
        #Defines LLM Client for custom genetic operators
        api_key = os.environ.get("TOGETHER_AI") #Uses the TogetherAI API

        if api_key is None:
            raise Exception("Together API key not found")

        client = Together(api_key=api_key)

        print("LLM Setup")
        return client
    
    def setupDaytona(self):
        daytonaClient = Daytona()
        sandbox = daytonaClient.create()
        
        #Install DEAP
        sandbox.process.exec("python -m pip install deap==1.4.1")
        print("DEAP installed")

        #Provide functions for pset
        with open("gp_primitives.py", "rb") as f:
            content = f.read()
            sandbox.fs.upload_file(content, "gp_primitives.py")
        print("Primitives copied")

        print("Sandbox initialised")

        return sandbox

    def evaluateIndividual(self, individual, points):
        #TODO: This is temporary for this specific function
        #Defines the fitness function for an individual 
        func = self.toolbox.compile(expr=individual) #Transform the tree expression in a callable function

        # Evaluate the mean squared error between the expression and the real function : x**4 + x**3 + x**2 + x
        sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
        return math.fsum(sqerrors) / len(points),

    def adaptive_operator_stats(self):
        stats = {}
        stats["crossover_redesigns"] = self.custom_crossover.total_num_redesigns
        stats["mutation_redesigns"] = self.mutator.total_num_redesigns

        #Adds extra generation at start with no value
        self.fitness_improvements[0] = np.nan
        stats["fitness_improvements"] = self.fitness_improvements

        #Tracks which generations we redesign the operators on
        stats["redesign_generations"] = self.redesign_generations

        return stats

    def runSimpleEA(self):
        self.reset_state()

        # Run GP - Simple Evolutionary Algorithm
        pop, log = algorithms.eaSimple(self.pop, self.toolbox, 0.5, 0.1, 40, stats=self.mstats,
                                    halloffame=self.hof, verbose=True)
        
        return pop, log, self.hof
        
    def check_stagnation(self, current_fitness, gen_num):
        #There has been an improvement
        self.gens_since_improvement += 1

        #Updates statistics
        self.fitness_improvements.append(self.prev_avg_fitness - current_fitness)

        if current_fitness < self.prev_avg_fitness:
            self.prev_avg_fitness = current_fitness
            self.gens_since_improvement = 0
        #There has not been an improvement, but less than k generations have surpassed
        elif current_fitness >= self.prev_avg_fitness and self.gens_since_improvement < self.k:
            self.gens_since_improvement += 1
        #There has not been an improvement in k generations
        elif current_fitness >= self.prev_avg_fitness and self.gens_since_improvement >= self.k:
            print("Stagnating.... Redesigning...")
            self.custom_crossover.redesign_operator()
            self.mutator.redesign_operator()
            self.redesign_generations.append(gen_num)
            self.gens_since_improvement = 0
        else:
            raise Exception("Error tracking fitnesses")
    
    def runDynamicEA(self, cxpb=0.5, mutpb=0.1, ngen=40, verbose=True):
        self.reset_state()

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (self.mstats.fields if self.mstats else [])

        #Tracks statistics for LLM
        history = {
            "avg_fitness": [],
            "max_fitness": [],
            "min_fitness": [],
            "avg_size": [],
            "max_size": [],
            "min_size": []
        }

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if self.hof is not None:
            self.hof.update(self.pop)

        record = self.mstats.compile(self.pop) if self.mstats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            offspring = self.toolbox.select(self.pop, len(self.pop))

            # Vary the pool of individuals
            #Attempt to create offspring - redesigning genetic operators if needed
            offspring = algorithms.varAnd(offspring, self.toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if self.hof is not None:
                self.hof.update(offspring)

            # Replace the current population by the offspring
            self.pop[:] = offspring

            # Append the current generation statistics to the logbook
            record = self.mstats.compile(self.pop) if self.mstats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

            avg_fitness = record['fitness']['avg']

            history["avg_fitness"].append(float(record["fitness"]["avg"]))
            history["max_fitness"].append(float(record["fitness"]["max"]))
            history["min_fitness"].append(float(record["fitness"]["min"]))
            history["avg_size"].append(float(record["size"]["avg"]))
            history["max_size"].append(float(record["size"]["max"]))
            history["min_size"].append(float(record["size"]["min"]))

            # Updates LLM prompt with updated logbook
            self.mutator.update_llm_prompt(history)
            self.custom_crossover.update_llm_prompt(history)

            #Solution found - early stopping
            if record["fitness"]["min"] < 0.00001:
                break

            #Each generation, reset the number of local skips each operator is allowed
            self.mutator.local_skips = 0
            self.custom_crossover.local_skips = 0

            #Resets number of attempts
            self.mutator.num_retries = 0
            self.custom_crossover.num_retries = 0

            self.check_stagnation(avg_fitness, gen)

        ao_stats = self.adaptive_operator_stats()
        return self.pop, logbook, self.hof, ao_stats
    
    def shutdown_sandbox(self):
        self.sandbox.delete()