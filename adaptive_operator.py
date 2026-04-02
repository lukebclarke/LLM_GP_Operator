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
from util import load_module

import textwrap 
import os
import sys
import inspect
import traceback
import time

class MaximumNumberRetries(Exception):
    """Exception raised when maximum number of attempts to redesign operator has been reached

    Attributes:
        message -- Explanation of the error
    """

    def __init__(self, num_parents):
        super().__init__(f"Maximum Number of Retries for Operator: {num_parents} Parents")

class AdaptiveOperator():
    def __init__(self, client, sandbox, pset, toolbox, custom_creator, num_parents, num_offspring, max_num_retries=5, max_local_skips=5, model="Qwen/Qwen3-Coder-Next-FP8"):
        self.llm_client = client
        self.llm_model = model

        #Problem Definition
        self.pset = pset
        self.toolbox = toolbox
        self.creator = custom_creator
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
        self.local_skips = 0

        self.total_num_redesigns = 0
        self.timeout=40
        self.max_timeout_retries = 10

        self.prev_design = None

        #Enables us to import operator designs stored in temp folder 
        sys.path.append('/temp')
    
    #TODO: Define custom exception classes, and add to doc strings

    def reset_operator(self):
        self.operator_design = None 
        self.current_operator_module = None
        self.operator_design_validated = False
        self.llm_prompt = None
        self.num_retries = 0
        self.local_skips = 0
        self.total_num_redesigns = 0

    def remove_design(self):
        self.operator_design = None 
        self.current_operator_module = None
        self.operator_design_validated = False
        self.local_skips = 0

    def load_operator_module(self, module_name, file_name):
        with open(f"temp/{file_name}.py", "w") as f:
            #Ensures imports are present
            f.write("from deap import base, creator, tools, gp\n")
            f.write("import random\n")
            f.write(self.operator_design)

        self.current_operator_module = load_module(module_name, f"temp/{file_name}.py")

    def update_llm_prompt(self, history):
        #LLMs can deal with JSON
        formatted_history = json.dumps(history, indent=2)
        self.llm_prompt = self.original_llm_prompt.replace("INSERT_LOGBOOK_HERE", str(formatted_history))

    def validate_individual(self, individual):
        try:
            h = individual.height
            return True
        except:
            return False

    def clean_individual(self, individual):
        #Unwraps tuple (if applicable)
        if isinstance(individual, tuple) and len(individual) == 1:
            individual = list(individual)[0]

        #Converts to individual
        if not isinstance(individual, creator.Individual):
            individual = creator.Individual(individual)

        return individual
    
    def llm_standard_model(self):
        for i in range(self.max_timeout_retries):
            try:
                results = {
                    "code": None,
                    "exception": False
                }

                def get_llm_response():
                    #Uploads files to sandbox
                    try:
                        response = self.llm_client.chat.completions.create(
                        model=self.llm_model,
                        temperature=0.80,
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

                #Uses threads to implement timeout
                t = threading.Thread(target=get_llm_response)
                t.start()
                t.join(self.timeout)

                if t.is_alive() or results["exception"] == True:
                    results["exception"] = False
                    raise Exception

                return results["code"]

            except Exception:
                time.sleep(1)
    
    def llm_reasoning_model(self):
        response = self.llm_client.chat.completions.create(
                model="MiniMaxAI/MiniMax-M2.5",
                temperature=0.80,
                messages=[
                {"role": "system", "content": "You provide Python code to be directly executed out-of-the-box. Return only raw Python code, do not include any additional text/explanations in your response."},
                {
                    "role": "user",
                    "content": self.llm_prompt
                }
                ],
                stream=False,
                max_tokens=2000,
            )
                
        code = response.choices[0].message.content
        
        return code

    def redesign_operator(self):
        self.remove_design()
        self.total_num_redesigns += 1
        self.local_skips = 0

        #TODO: Create a counter of how many time it retries
        code = ""
        #Keeps generating until correct format is produced
        while self.num_retries <= self.max_num_retries:
            self.num_retries += 1

            code = self.llm_standard_model()

            #Must contain the operator function
            if ("def crossover_individuals(" in code) or ("def mutate_individual(" in code):
                break

        if self.num_retries > self.max_num_retries:
            raise MaximumNumberRetries(self.num_parents)

        #Saves the resulting function - can be reaccessed
        self.operator_design = clean_llm_output(code)
        self.operator_design_validated = False #TODO: Merge these variables
        self.current_operator_module = None

    def llm_custom_operator_daytona(self, individuals):
        """Used to execute LLM-generated code using the Daytona sandbox. Verifies that the code is able to execute successfully, and that it is trusted.

        Args:
            individuals ([gp.Individual]): A list of the parent individuals
            llm_client (Together): The LLM client used to redesign the operator
            sandbox (Daytona): The Daytona sandbox used to execute the LLM-generated design

        Raises:
            Exception: TODO

        Returns:
            [gp.Individual]: The offspring of the operation
        """

        #Pickles objects - enables transfer to sandbox environment
        for i in range(self.num_parents):
            pickle_object(individuals[i], f"individual{i}")
        pickle_object(self.pset, "pset") #TODO: Upload at start

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

                        with open("temp/pset.pkl", "rb") as f:
                            content = f.read()
                            self.sandbox.fs.upload_file(content, "pset.pkl")

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

                            if not self.validate_individual(results["offspring"][i]):
                                raise Exception("Invalid offspring generated")

                        self.operator_design_validated = True
                        self.current_operator_module = None


                        log = self.sandbox.fs.download_file("error.txt")
                        # print(log.decode("utf-8")) #Prints error log
                    except Exception:
                        results["exception"] = True

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
    
        #If exceed maximum number of attempts, just use previous design
        if self.prev_design != None:
            self.operator_design_validated = True
            self.current_operator_module = self.prev_design
            return self.llm_custom_operator_locally(individuals)
        #If no previous design, just use uniform mutation
        elif self.prev_design == None and self.num_parents == 1:
            ind = gp.mutUniform(individuals[0], expr=self.toolbox.expr_mut, pset=self.pset)
            return ind
        #If no previous design, just use one point crossover
        elif self.prev_design == None and self.num_parents == 2:
            ind1, ind2 = gp.cxOnePoint(individuals[0], individuals[1])
            return ind1, ind2
        else:
            raise Exception("Fatal error")
    
    def apply_operator(self, individuals):
        """This method should be overwritten in the child class"""
        return individuals
    
    def llm_custom_operator_locally(self, individuals):
        """Used to execute LLM-generated code locally. Used for trusted code.

        Args:
            individuals ([gp.Individual]):  A list of the parent individuals
            llm_client (Together): The LLM client used to redesign the operator

        Raises:
            Exception: TODO

        Returns:
            [gp.Individual]: A list of the offspring of the operation
        """
        #TODO: Wrap in try except - redesign if crashes
        #TODO: Do we need to pass LLM_Client?

        #Ensure that the Python design is saved locally
        if self.current_operator_module == None:
            wrapper_text = self.local_wrapper.replace("INSERT_METHOD_DEFINITION_HERE", self.operator_design)

            self.current_operator_module = compile(wrapper_text, f"operator_{self.num_parents}", "exec")

            #TODO: Temp for testing
            with open("temp/testing_current_local_design.py", "w") as f:
                f.write(f"#Time execution: {time.time()}\n")
                f.write(self.operator_design)

        #Attempt to apply operator locally TODO - Used for testing purposes, can remove
        if self.current_operator_module != None:
            try:
                offspring = self.apply_operator(individuals)

                #Ensure correct types
                for i in range(len(offspring)):
                    offspring[i] = self.clean_individual(offspring[i])

                    if not self.validate_individual(offspring[i]):
                        raise Exception("Invalid offspring generated")

                #TODO: Reset num_retries at end of generation
                #Only once the module has been operated locally, do we accept the design
                self.num_retries = 0

                # self.prev_design = self.current_operator_module

                return offspring
            
            #Redesign if code is unable to execute locally
            except Exception as e:
                #Get new individuals (TODO)

                self.local_skips += 1
                if self.local_skips >= self.max_local_skips:
                    print("Maximum number of local skips exceeded, redesigning operator...")
                    self.redesign_operator()

                #If operator doesn't work, return the original individuals 
                return individuals
            
        else:
            raise Exception("No module loaded for custom operator") #TODO: Sort out error handling
