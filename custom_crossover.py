from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from daytona import Daytona
from daytona import CodeRunParams

from adaptive_operator import AdaptiveOperator

import textwrap 
import os
import sys
import inspect

class CustomCrossover(AdaptiveOperator):
    def __init__(self, client, sandbox, pset, toolbox, model="Qwen/Qwen3-Coder-Next-FP8", max_num_retries=5):
        #Crossover operators have 2 parents and 2 offspring
        super().__init__(client, sandbox, pset, toolbox, num_parents=2, num_offspring=2, max_num_retries=max_num_retries, model=model)

        #Wrapper for code
        with open("docs/crossover_wrapper.txt", "r") as f:
            self.daytona_wrapper = f.read()

        #Gets LLM Prompt from file
        with open("docs/LLMPromptCrossover.txt", "r") as f:
            self.original_llm_prompt = f.read()
        self.llm_prompt = None

        #Enables us to import crossover design stored in temp folder 
        sys.path.append('/temp')

    def apply_operator(self, individuals):
        offspring = self.current_operator_module.crossover_individuals(individuals[0], individuals[1], self.pset)
        return [offspring[0], offspring[1]]

    def crossover(self, individual1, individual2):

        print(f"CURRENT CROSS1 TYPE: {type(individual1)}")
        print(f"CURRENT CROSS1: {individual1}")

        print(f"CURRENT CROSS2 TYPE: {type(individual2)}")
        print(f"CURRENT CROSS: {individual2}")

        #By default, use one point crossover
        if self.operator_design == None:
            ind1, ind2 = gp.cxOnePoint(individual1, individual2)
            print(f"OFFSPRING 1 TYPE STANDARD: {type(ind1)}")
            print(f"OFFSPRING 1: {ind1}")
            return ind1, ind2
        #Validates the design by using Daytona to execute the code
        elif self.operator_design != None and self.operator_design_validated == False:
            offspring = self.llm_custom_operator_daytona([individual1, individual2])
            print(f"OFFSPRING 1 TYPE: {type(offspring[0])}")
            print(f"OFFSPRING 1: {offspring[0]}")
            print(f"OFFSPRING 2 TYPE: {type(offspring[1])}")
            print(f"OFFSPRING 2: {offspring[1]}")
            return offspring[0], offspring[1]
        #If the design has already been validated, can execute locally
        elif self.operator_design != None and self.operator_design_validated == True:
            #Ensure design is saved locally
            offspring = self.llm_custom_operator_locally([individual1, individual2])
            print(f"OFFSPRING 1 TYPE: {type(offspring[0])}")
            print(f"OFFSPRING 1: {offspring[0]}")
            print(f"OFFSPRING 2 TYPE: {type(offspring[1])}")
            print(f"OFFSPRING 2: {offspring[1]}")
            return offspring[0], offspring[1]

