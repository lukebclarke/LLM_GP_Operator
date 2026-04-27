from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from daytona import Daytona
from daytona import CodeRunParams

import json
import random
import pickle
import threading

from util import clean_llm_output
from util import pickle_object
from util import unpickle_daytona_file

import textwrap 
import os
import sys
import inspect
import traceback
import time

import logging
import google.cloud.logging

class MaximumNumberRetries(Exception):
    """Exception raised when maximum number of attempts to redesign operator has been reached

    Attributes:
        message -- Explanation of the error
    """

    def __init__(self, num_parents):
        super().__init__(f"Maximum Number of Retries for Operator: {num_parents} Parents")

class BaseOperator():
    def __init__(self, client, sandbox, pset, toolbox, num_parents, num_offspring, default_temperature=0.3, temperature_alpha=0.1, max_num_retries=5, max_local_skips=5, model="Qwen/Qwen3-Coder-Next-FP8", reasoning_model=False):
        self.llm_client = client
        self.llm_model = model
        self.reasoning_model = reasoning_model

        #Problem Definition
        self.pset = pset
        self.toolbox = toolbox
        self.sandbox = sandbox

        #Number of times to retry generating LLM response 
        self.max_num_retries = max_num_retries 
        self.max_local_skips = max_local_skips

        #Setup custom operator
        self.operator_design = None 
        self.current_operator_module = None
        self.operator_design_validated = False
        self.daytona_wrapper = None
        self.local_wrapper = None
        self.original_llm_prompt = None
        self.llm_prompt = None
        self.num_parents = num_parents
        self.num_offspring = num_offspring

        self.num_retries = 0

        self.total_num_redesigns = 0
        self.timeout=40
        self.max_timeout_retries = 10

        #Tracks how many redesigns are effective/ineffective
        self.effective_redesigns = 0

        #Temperature Self-Adaptation
        self.temperature = default_temperature
        self.temperature_alpha = temperature_alpha

        #Tracks previous design
        self.prev_design = None

        #Statistics for current desing
        self.total_operator_skips = 0
        self.total_operator_evals = 0

        #Enables us to import operator designs stored in temp folder 
        sys.path.append('/temp')
    
    def update_llm_prompt(self, history, example_design, best_solution):
        """Adds additional information about algortihm progress to LLM prompt

        Args:
            history (dict): The logbook for the run
        """
        #Specify what to include in prompt
        filtered_history = {
            "avg_fitness": history["avg_fitness"],
            "min_fitness": history["min_fitness"],
            "avg_size": history["avg_size"]
        }

        #LLMs can deal with JSON
        formatted_history = json.dumps(filtered_history, indent=2)

        #Setup prompt by inserting logbook, example method, and current best solution
        self.llm_prompt = self.original_llm_prompt.replace("INSERT_LOGBOOK_HERE", str(formatted_history))
        self.llm_prompt = self.llm_prompt.replace("INSERT_BEST_SOLUTION_HERE", str(best_solution))
        self.llm_prompt = self.llm_prompt.replace("INSERT_EXAMPLE_METHOD", example_design)

    def self_adapt_temperature(self):
        """Increases the temperature to improve diversity"""
        self.temperature = self.temperature + self.temperature_alpha

        #Prevents temperature exceeding 1
        self.temperature = min(1, self.temperature)

    def validate_individual(self, individual):
        """Ensures that an individual is valid (i.e. compatible with DEAP)

        Args:
            individual (gp.Individual): The individual to check

        Returns:
            bool: True if the individual is compatible with DEAP
        """
        try:
            func = self.toolbox.compile(expr=individual)
            height = individual.height
            return True
        except Exception as e:
            print(e)
            return False

    def clean_individual(self, individual):
        """Ensures individual is in the correct format

        Args:
            individual (gp.Individual): Individual to clean

        Returns:
            gp.Individual: The individual in the correct format
        """
        #Unwraps tuple (if applicable)
        if isinstance(individual, tuple) and len(individual) == 1:
            individual = list(individual)[0]

        #Converts to individual
        if not isinstance(individual, creator.Individual):
            individual = creator.Individual(individual)

        return individual
    
    def get_standard_llm_response(self, results):
        """Gets response from LLM. To be used in threading context

        Args:
            results (dict): The dictionary of results to update. Includes 'code' and 'exception' entries.
        """
        try:
            response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            temperature=self.temperature,
            messages=[
            {"role": "system", "content": "You provide Python code to be directly executed out-of-the-box. Return only raw Python code, do not include any additional text/explanations in your response."},
            {
                "role": "user",
                "content": self.llm_prompt
            }
            ],
        )
            
            code = response.choices[0].message.content
        
            results["code"] = code
            results["exception"] = False
            
        except Exception:
            results["exception"] = True

    def get_reasoning_llm_response(self, results):
        """Gets response from LLM. To be used in threading context

        Args:
            results (dict): The dictionary of results to update. Includes 'code' and 'exception' entries.
        """
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                temperature=self.temperature,
                messages=[
                {"role": "system", "content": "You provide Python code to be directly executed out-of-the-box. Return only raw Python code, do not include any additional text/explanations in your response."},
                {
                    "role": "user",
                    "content": self.llm_prompt,
                }
                ],
                reasoning={"enabled": False},
                max_tokens=2000,
            )
            code = response.choices[0].message.content

            results["code"] = code
            results["exception"] = False
            
        except Exception:
            results["exception"] = True
    
    def prompt_llm(self):
        """Prompts a standard, non-reasoning LLM model for an operator design.

        Returns:
            string: The generated operator design
        """
        for _ in range(self.max_timeout_retries):
            try:
                #Results are defined in dictionary so they can be passed to threads
                results = {
                    "code": None,
                    "exception": False
                }

                #Uses threads to implement timeout
                if self.reasoning_model:
                    t = threading.Thread(target=self.get_reasoning_llm_response, args=(results,))
                else:
                    t = threading.Thread(target=self.get_standard_llm_response, args=(results,))

                t.start()
                t.join(self.timeout)

                #If thread is still running after timeout, terminate and try again
                if t.is_alive() or results["exception"] == True:
                    results["exception"] = False
                    raise Exception
                                
                return results["code"]
        
            except Exception:
                #Wait before retrying
                time.sleep(1)
    
    def redesign_operator(self):
        """Redesigns the current operator using LLMs

        Raises:
            MaximumNumberRetries: Raised if we attempt more than the maximum specified number of retries
        """
        #Resets state of operator
        self.operator_design = None 
        self.current_operator_module = None
        self.operator_design_validated = False

        #Resets statistics for operator
        self.total_operator_skips = 0
        self.total_operator_evals = 0

        #Increments counter
        self.total_num_redesigns += 1

        code = ""
        #Keeps generating until correct format is produced
        while self.num_retries <= self.max_num_retries:
            self.num_retries += 1

            code = self.prompt_llm()

            #Must contain the operator function
            if ("def crossover_individuals(" in code) or ("def mutate_individual(" in code):
                break

        if self.num_retries > self.max_num_retries:
            raise MaximumNumberRetries(self.num_parents)

        #Saves the resulting function - can be reaccessed
        self.operator_design = clean_llm_output(code)
        self.operator_design_validated = False
        self.current_operator_module = None

    def llm_custom_operator_daytona(self, individuals):
        """Used to execute LLM-generated code using the Daytona sandbox. Verifies that the code is able to execute successfully, and that it is trusted.

        Args:
            individuals ([gp.Individual]): A list of the parent individuals
            llm_client (Together): The LLM client used to redesign the operator
            sandbox (Daytona): The Daytona sandbox used to execute the LLM-generated design

        Raises:
            MaximumNumberRetries: Raised if we attempt more than the maximum specified number of retries

        Returns:
            [gp.Individual]: The offspring of the operation
        """
        #Pickles objects - enables transfer to sandbox environment
        for i in range(self.num_parents):
            pickle_object(individuals[i], f"individual{i}")
        
        #Uploads files (uses threads so we can reattempt after timeout)
        for i in range(self.max_timeout_retries):
            results = {"exception": False}
            try:

                def upload_files():
                    #Uploads files to sandbox
                    try:
                        for i in range(self.num_parents):
                            with open(f"temp/individual{i}.pkl", "rb") as f:
                                content = f.read()
                                self.sandbox.fs.upload_file(content, f"individual{i}.pkl")

                    except Exception:
                        results["exception"] = True

                #Uses threads to implement timeout
                t = threading.Thread(target=upload_files)
                t.start()
                t.join(self.timeout)

                if t.is_alive() or results["exception"] == True:
                    results["exception"] = False
                    raise Exception

                break

            except Exception:
                time.sleep(1)
            
        #Attempts to execute - redesigns operator if fails
        while self.num_retries < self.max_num_retries: 
            print(f"Number of redesigns: {self.num_retries}")
            #Inserts LLM-generated function into full operator code
            operator_code = textwrap.indent(self.operator_design, "    ")
            wrapper_text = self.daytona_wrapper.replace("INSERT_METHOD_DEFINITION_HERE", operator_code)

            print(wrapper_text)

            try:
                results = {"offspring": [],
                           "exception": None}
                
                def execute_llm_code():
                    try:
                        compile(wrapper_text, "<sandbox>", "exec")
                    
                        #Execute the code in the sandbox
                        response = self.sandbox.process.code_run(wrapper_text)

                        #Must convert the pickled results back to desired form
                        for i in range(self.num_offspring):
                            results["offspring"].append(unpickle_daytona_file(f"offspring{i}", self.sandbox)) 
                            results["offspring"][i] = self.clean_individual(results["offspring"][i])

                            print("OFFSPRING")
                            print(results["offspring"])

                            if not self.validate_individual(results["offspring"][i]):
                                raise Exception("Invalid offspring generated")
                            
                        self.operator_design_validated = True
                        self.current_operator_module = None

                    except Exception as e:
                        results["exception"] = True
                        print(e)

                #Uses threads to implement timeout
                t = threading.Thread(target=execute_llm_code)
                t.start()
                t.join(self.timeout)

                if t.is_alive():
                    raise TimeoutError("Operation timed out")
                
                if results["exception"] == True:
                    raise Exception("Invalid offspring generated")

                return results["offspring"]
            
            except TimeoutError as e:
                self.num_retries += 1
                continue

            except Exception as e:
                #If an error occurs, attempt to redesign the LLM function
                self.redesign_operator()

        print("Maximum number of attempts exceeded...")
        raise MaximumNumberRetries(self.num_parents)
    
    def apply_operator(self, individuals):
        """This method should be overwritten in the child class"""
        return individuals
    
    def llm_custom_operator_locally(self, individuals):
        """Used to execute LLM-generated code locally. Used for trusted code.

        Args:
            individuals ([gp.Individual]):  A list of the parent individuals
            llm_client (Together): The LLM client used to redesign the operator

        Returns:
            [gp.Individual]: A list of the offspring of the operation
        """
        #Ensure that the Python design is saved locally
        if self.current_operator_module == None:
            wrapper_text = self.local_wrapper.replace("INSERT_METHOD_DEFINITION_HERE", self.operator_design)

            self.current_operator_module = compile(wrapper_text, f"operator_{self.num_parents}", "exec")

        #Attempt to apply operator locally
        try:
            offspring = self.apply_operator(individuals)
            
            #Ensure correct types
            for i in range(len(offspring)):
                offspring[i] = self.clean_individual(offspring[i])

                if not self.validate_individual(offspring[i]):
                    raise Exception("Invalid offspring generated")

            #Only once the module has been operated locally, do we accept the design
            self.num_retries = 0
            self.total_operator_evals += 1

            return offspring
        
        #Redesign if code is unable to execute locally
        except Exception as e:
            self.total_operator_skips += 1
            print(e)
            if self.total_operator_skips >= self.max_local_skips:
                print("Maximum number of local skips exceeded, redesigning operator...")
                self.redesign_operator()

            #If operator doesn't work, return the original individuals 
            return individuals

        except TimeoutError as e:
            print("Locally applied operator timeout... redesigning")
            self.redesign_operator()
