from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from daytona import Daytona
from daytona import CodeRunParams

import textwrap 
import os
import sys
import inspect

from adaptive_operator import AdaptiveOperator

class CustomMutate(AdaptiveOperator):
    def __init__(self, client, sandbox, pset, toolbox, model="Qwen/Qwen3-Coder-Next-FP8", max_num_retries=5):
        #Mutation operators have 2 parents and 2 offspring
        super().__init__(client, sandbox, pset, toolbox, num_parents=1, num_offspring=1, max_num_retries=max_num_retries, model=model)

        #Wrapper for code
        with open("docs/mutation_wrapper.txt", "r") as f:
            self.daytona_wrapper = f.read()

        #Gets LLM Prompt from file
        with open("docs/LLMPromptMutation.txt", "r") as f:
            self.original_llm_prompt = f.read()
        self.llm_prompt = None

        #Enables us to import mutation design stored in temp folder 
        sys.path.append('/temp')

    def apply_operator(self, individuals):
        ind = individuals[0]
        offspring = self.current_operator_module.mutate_individual(ind, self.pset)

        #Ensures in correct format
        if isinstance(offspring, tuple) and len(offspring)==1:
            return [offspring[0]]
        else:
            return [offspring]

    def mutate(self, individual):
        #By default, use uniform mutation
        if self.operator_design == None:
            ind = gp.mutUniform(individual, expr=self.toolbox.expr_mut, pset=self.pset)
            return ind
        #Validates the design by using Daytona to execute the code
        elif self.operator_design != None and self.operator_design_validated == False:
            offspring = self.llm_custom_operator_daytona([individual])

            return offspring[0],
        #If the design has already been validated at least 3 times, can execute locally
        elif self.operator_design != None and self.operator_design_validated == True:
            print("Testing locally")
            #Ensure design is saved locally
            offspring = self.llm_custom_operator_locally([individual])[0],

            return offspring[0],
