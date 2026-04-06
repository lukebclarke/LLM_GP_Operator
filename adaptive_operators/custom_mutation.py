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

from adaptive_operators.base_operator import BaseOperator

class CustomMutation(BaseOperator):
    def __init__(self, client, sandbox, pset, toolbox, model="Qwen/Qwen3-Coder-Next-FP8", max_local_skips=5, max_num_retries=5, default_temperature=0.3, temperature_alpha=0.1):
        #Mutation operators have 2 parents and 2 offspring
        super().__init__(client, sandbox, pset, toolbox, num_parents=1, num_offspring=1, max_num_retries=max_num_retries, max_local_skips=max_local_skips, model=model, default_temperature=default_temperature, temperature_alpha=temperature_alpha)

        #Wrapper for code
        with open("docs/mutation_remote_wrapper.txt", "r") as f:
            self.daytona_wrapper = f.read()

        with open("docs/mutation_local_wrapper.txt", "r") as f:
            self.local_wrapper = f.read()

        #Gets LLM Prompt from file
        with open("docs/LLMPromptMutation.txt", "r") as f:
            self.original_llm_prompt = f.read()
        self.llm_prompt = None

        #Enables us to import mutation design stored in temp folder 
        sys.path.append('/temp')

    def apply_operator(self, individuals):
        """Applies custom mutation locally to individuals

        Args:
            individuals (list): List of gp.Individual objects

        Returns:
            list: List of the offspring of type gp.Individual
        """
        #Sets up environment
        global_env = {
            "individual": individuals[0],
            "pset": self.pset, 
            "creator": creator
        }

        #Executes compiled code
        exec(self.current_operator_module, global_env, global_env)

        offspring = global_env["result"]

        #Ensures in correct format
        if isinstance(offspring, tuple) and len(offspring)==1:
            return [offspring[0]]
        else:
            return [offspring]

    def mutate(self, individual):
        """Determines which mutation operator to apply, and applies it

        Args:
            individual (gp.Individual): The parent to mutate

        Returns:
            (gp.Individual,): The offspring of the mutation operator
        """
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
            #Ensure design is saved locally
            offspring = self.llm_custom_operator_locally([individual])

            return offspring[0],
