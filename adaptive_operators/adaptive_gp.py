import operator
import math
import random
import os
import numpy as np
import time
import threading

import numpy

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from util import pickle_object

#Visualisation
import pygraphviz as pgv

#LLM
from google import genai
from google.genai import types
from together import Together
from daytona import Daytona
import subprocess

#Files
from adaptive_operators.custom_mutation import CustomMutation
from adaptive_operators.custom_crossover import CustomCrossover

class AdaptiveGP():
    def __init__(self, n, pset, toolbox, client, sandbox, custom_mutate, custom_crossover, X, Y, k=2, self_adapt_req=4, timeout=20):
        self.n = n
        self.pset = pset
        self.toolbox = toolbox
        self.X = X
        self.Y = Y
        self.timeout = timeout

        self.custom_mutate = custom_mutate
        self.custom_crossover = custom_crossover
        self.client = client
        self.sandbox = sandbox

        #Initialises population
        self.pop = self.toolbox.population(n=n)
        self.hof = tools.HallOfFame(1) #We track 1 best solution

        #Track statistics
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        self.mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        self.mstats.register("avg", numpy.mean)
        self.mstats.register("std", numpy.std)
        self.mstats.register("min", numpy.min)
        self.mstats.register("max", numpy.max)
        self.fitness_improvements = []
        self.redesign_generations = []

        #Operator design statistics
        self.mutation_designs = []
        self.crossover_designs = []

        self.k = k #Redesign algorithm if there has no improvement in fitness for K generations
        self.gens_since_redesign = 0
        self.prev_min_fitness = np.inf

        #Self-adaptation variables
        self.gens_since_improvement = 0 #Used for self-adaptation
        self.self_adapt_req = self_adapt_req

    def get_stats(self):
        """Returns stats about the algorithm instance

        Returns:
            dict: Statistics including 'crossover_redesigns', 'mutation_redesigns', 'fitness_improvements' and 'redesign_generations'
        """
        stats = {}
        stats["num_crossover_redesigns"] = self.custom_crossover.total_num_redesigns
        stats["num_mutation_redesigns"] = self.custom_mutate.total_num_redesigns

        #Adds extra generation at start with no value
        self.fitness_improvements[0] = np.nan
        stats["fitness_improvements"] = self.fitness_improvements

        #Tracks which generations we redesign the operators on
        stats["redesign_generations"] = self.redesign_generations

        #Tracks all operator designs
        stats["crossover_designs"] = self.crossover_designs
        stats["mutation_designs"] = self.mutation_designs

        return stats

    def update_operator_history(self, current_gen, logbook):
        """Updates the record of previous operator designs, associating a score with the current operators

        Args:
            current_gen (int): The generation number we are currently on
            logbook (tools.Logbook): The logbook containing the history of the algorithm
        """
        current_mutator_design = self.custom_mutate.operator_design
        current_crossover_design = self.custom_crossover.operator_design

        #Tracks statistics for operator design, used to calculate score
        mutation_stats = {
            "success_rate": None,
            "min_fitness_improv": None,
            "avg_fitness_improv": None,
            "score": None
        }

        crossover_stats = {
            "success_rate": None,
            "min_fitness_improv": None,
            "avg_fitness_improv": None,
            "score": None
        }

        #Success Rate for Mutation
        if self.custom_mutate.total_operator_evals == 0:
            mutator_success_rate = 0
        else:
            mutator_success_rate = (self.custom_mutate.total_operator_evals - self.custom_mutate.total_operator_skips) / (self.custom_mutate.total_operator_evals)
        
        #Success Rate for Crossover
        if self.custom_crossover.total_operator_evals == 0:
            crossover_success_rate = 0
        else:
            crossover_success_rate = (self.custom_crossover.total_operator_evals - self.custom_crossover.total_operator_skips) / (self.custom_crossover.total_operator_evals)
        

        first_gen = self.redesign_generations[-1]

        #Fitness improvements (average and minimum)
        min_fitness = logbook.chapters["fitness"].select("min")
        avg_fitness = logbook.chapters["fitness"].select("avg")

        #Finds percent improvement per generation
        percent_improvement_min_fitness = []
        percent_improvement_avg_fitness = []
        for gen in range(first_gen-1, current_gen-1):
            #Finds improvement in avg fitness from last generation
            avg_improvement = avg_fitness[gen] - avg_fitness[gen+1]
            avg_improvement = max(0.2, avg_improvement) #Can't go below 0.2
            percent_improvement_avg_fitness.append((avg_improvement) / abs(avg_fitness[gen]))

            #Finds improvement in minimum fitness from last generation
            min_improvement = min_fitness[gen] - min_fitness[gen+1]
            min_improvement = max(0.2, min_improvement) #Can't go below 0.2
            percent_improvement_min_fitness.append((min_improvement) / abs(min_fitness[gen]))

        #Calculates 'fitness' score of operator design
        mean_min_fitness_improv = sum(percent_improvement_min_fitness) / len(percent_improvement_min_fitness)
        mean_avg_fitness_improv = sum(percent_improvement_avg_fitness) / len(percent_improvement_avg_fitness)

        #Updates statistics
        mutation_stats["success_rate"] = mutator_success_rate
        crossover_stats["success_rate"] = crossover_success_rate

        mutation_stats["min_fitness_improv"] = mean_min_fitness_improv
        crossover_stats["min_fitness_improv"] = mean_min_fitness_improv

        mutation_stats["avg_fitness_improv"] = mean_avg_fitness_improv
        crossover_stats["avg_fitness_improv"] = mean_avg_fitness_improv

        mutation_stats["score"] = (mutator_success_rate * mean_min_fitness_improv * mean_avg_fitness_improv)
        crossover_stats["score"] = (crossover_success_rate * mean_min_fitness_improv * mean_avg_fitness_improv)

        #Adds design + corresponding score to history
        self.mutation_designs.append((current_mutator_design, mutation_stats))
        self.crossover_designs.append((current_crossover_design, crossover_stats))

    def get_default_operator_designs(self):
        """
        Loads default operator designs from text file to be used as an example for LLM
        """
         #Loads default crossover design
        f = open("docs/default_crossover_design.txt")
        crossover_design = f.read()
        f.close()

        f = open("docs/default_mutation_design.txt")
        mutation_design = f.read()
        f.close()

        return crossover_design, mutation_design

    def get_operator_design(self):
        """Proportionally selects mutation and crossover designs to feed back into LLM, based on their performance

        Returns:
            string, string: Mutation and crossover designs, respectively 
        """
        #Return default designs if the operators have not yet been redesigns
        if not self.redesign_generations:
            return self.get_default_operator_designs()

        #Split lists into separate lists of designs and statistics
        mutation_designs, mutation_stats = zip(*self.mutation_designs)
        crossover_designs, crossover_stats = zip(*self.crossover_designs)

        #Finds raw scores
        mutation_scores = [stats["score"] for stats in mutation_stats]
        crossover_scores = [stats["score"] for stats in mutation_stats]

        total_mut_scores = sum(mutation_scores)
        total_cross_scores = sum(crossover_scores)

        #Falls back to default designs if there are no successful operators
        if total_mut_scores or total_cross_scores <= 0:
            default_crossover, default_mutation = self.get_default_operator_designs() 
        
        #Chooses design proportionally based on associated operator score
        if total_mut_scores > 0:
            mutation_design = random.choices(mutation_designs, weights=mutation_scores, k=1)[0]
        else:
            mutation_design = default_mutation

        #Repeats for crossover operator
        if total_cross_scores > 0:
            crossover_design = random.choices(crossover_designs, weights=crossover_scores, k=1)[0]
        else:
            crossover_design = default_crossover

        return mutation_design, crossover_design

    def check_stagnation(self, current_fitness, gen_num, logbook, history):
        """Checks whether the design of the algorithm is stagnating, and redesigns if so

        Args:
            current_fitness (float): The current minimum fitness of the generation
            gen_num (int): Generation number

        Raises:
            Exception: Raised when fitnesses are not being tracked correctly
        """
        #Updates statistics
        self.fitness_improvements.append(self.prev_min_fitness - current_fitness)

        #There has been an improvement - continue evolution as normal
        if current_fitness < self.prev_min_fitness:
            self.prev_min_fitness = current_fitness
            self.gens_since_redesign = 0
            self.gens_since_improvement = 0
        #There has not been an improvement, but less than k generations have surpassed
        elif current_fitness >= self.prev_min_fitness and self.gens_since_redesign < self.k:
            self.gens_since_redesign += 1
            self.gens_since_improvement += 1
        #There has not been an improvement in k generations - redesign operator design
        elif current_fitness >= self.prev_min_fitness and self.gens_since_redesign >= self.k:
            print("Stagnating.... Redesigning...")
            #Adds operator design to history
            if len(self.redesign_generations) > 0:
                self.update_operator_history(gen_num, logbook)

            #Selects operator designs to use as example
            crossover_design, mutation_design = self.get_operator_design()

            #Updates prompt to include fitness history and example operator design
            self.custom_crossover.update_llm_prompt(history, crossover_design, self.hof[0])
            self.custom_mutate.update_llm_prompt(history, mutation_design, self.hof[0])

            #Redesigns both operators
            self.custom_crossover.redesign_operator()
            self.custom_mutate.redesign_operator()

            self.redesign_generations.append(gen_num)
            self.gens_since_redesign = 0
            self.gens_since_improvement += 1
        else:
            raise Exception("Error tracking fitnesses")
    
    def run_adaptive_ea(self, cxpb=0.8, mutpb=0.1, ngen=40, verbose=True):
        """Runs a standard evolutionary algorithm, redesigning genetic operators after periods of stagnation.

        Args:
            cxpb (float, optional): The probability of crossover. Defaults to 0.8.
            mutpb (float, optional): The probability of mutation. Defaults to 0.1.
            ngen (int, optional): The number of generations. Defaults to 40.
            verbose (bool, optional): Outputs current record of logbook at each generation. Defaults to True.

        Returns:
            list, Logbook, list, dict: The population, logbook, hall of fame individuals, and stats
        """
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

            min_fitness = record['fitness']['min']

            history["avg_fitness"].append(float(record["fitness"]["avg"]))
            history["max_fitness"].append(float(record["fitness"]["max"]))
            history["min_fitness"].append(float(record["fitness"]["min"]))
            history["avg_size"].append(float(record["size"]["avg"]))
            history["max_size"].append(float(record["size"]["max"]))
            history["min_size"].append(float(record["size"]["min"]))

            #Solution found - early stopping
            if record["fitness"]["min"] < 0.00001:
                break

            #Each generation, reset the number of local skips each operator is allowed
            self.custom_mutate.local_skips = 0
            self.custom_crossover.local_skips = 0

            #Resets number of attempts
            self.custom_mutate.num_retries = 0
            self.custom_crossover.num_retries = 0

            self.check_stagnation(min_fitness, gen, logbook, history)

            #Self-adapt temperature if there has been no improvement for a certain number of generations (even after redesigns)
            if self.gens_since_improvement >= self.self_adapt_req:
                self.custom_mutate.self_adapt_temperature()
                self.custom_crossover.self_adapt_temperature()
                self.gens_since_improvement = 0

        ao_stats = self.get_stats()
        
        return self.pop, logbook, self.hof, ao_stats
