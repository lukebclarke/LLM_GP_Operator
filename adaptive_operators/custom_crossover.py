from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from daytona import Daytona
from daytona import CodeRunParams

from adaptive_operators.base_operator import BaseOperator

import textwrap 
import os
import sys
import inspect

class CustomCrossover(BaseOperator):
    def __init__(self, client, sandbox, pset, toolbox, model="Qwen/Qwen3-Coder-Next-FP8", max_local_skips=5, max_num_retries=5):
        #Crossover operators have 2 parents and 2 offspring
        super().__init__(client, sandbox, pset, toolbox, num_parents=2, num_offspring=2, max_num_retries=max_num_retries, max_local_skips=max_local_skips, model=model)

        #Wrapper for code
        with open("docs/crossover_remote_wrapper.txt", "r") as f:
            self.daytona_wrapper = f.read()

        with open("docs/crossover_local_wrapper.txt", "r") as f:
            self.local_wrapper = f.read()

        #Gets LLM Prompt from file
        with open("docs/LLMPromptCrossover.txt", "r") as f:
            self.original_llm_prompt = f.read()
        self.llm_prompt = None

        #Enables us to import crossover design stored in temp folder 
        sys.path.append('/temp')

    def apply_operator(self, individuals):
        # offspring = self.current_operator_module.crossover_individuals(individuals[0], individuals[1], self.pset)

        #Sets up environment
        global_env = {
            "individual1": individuals[0],
            "individual2": individuals[1],
            "pset": self.pset, 
            "creator": creator
        }

        #Executes compiled code
        exec(self.current_operator_module, global_env, global_env)

        return [global_env["offspring1"], global_env["offspring2"]]

    def crossover(self, individual1, individual2):
        #By default, use one point crossover
        #TODO: REVERT THIS BACK
        if self.operator_design == None:
        # if True:
            ind1, ind2 = gp.cxOnePoint(individual1, individual2)
            return ind1, ind2
        #Validates the design by using Daytona to execute the code
        elif self.operator_design != None and self.operator_design_validated == False:
            offspring = self.llm_custom_operator_daytona([individual1, individual2])
            return offspring[0], offspring[1]
        #If the design has already been validated, can execute locally
        elif self.operator_design != None and self.operator_design_validated == True:
            #Ensure design is saved locally
            offspring = self.llm_custom_operator_locally([individual1, individual2])
            return offspring[0], offspring[1]

