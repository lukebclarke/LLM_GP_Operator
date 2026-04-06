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
    def __init__(self, client, sandbox, pset, toolbox, model="Qwen/Qwen3-Coder-Next-FP8", reasoning_model=False, max_local_skips=5, max_num_retries=5, default_temperature=0.3, temperature_alpha=0.1):
        #Crossover operators have 2 parents and 2 offspring
        super().__init__(client, sandbox, pset, toolbox, num_parents=2, num_offspring=2, max_num_retries=max_num_retries, default_temperature=default_temperature, temperature_alpha=temperature_alpha, max_local_skips=max_local_skips, model=model, reasoning_model=reasoning_model)

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
        """Applies custom crossover locally to individuals

        Args:
            individuals (list): List of gp.Individual objects

        Returns:
            list: List of the offspring of type gp.Individual
        """
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
        """Determines which crossover operator to apply, and applies it

        Args:
            individual1 (gp.Individual): The first parent
            individual2 (gp.Individual): The second parent

        Returns:
            gp.Individual, gp.Individual: The pair of offspring
        """
        #By default, use one point crossover
        if self.operator_design == None:
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

