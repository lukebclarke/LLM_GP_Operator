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

from util import clean_llm_output
from util import pickle_object
from util import unpickle_daytona_file
from util import load_module

import textwrap 
import os
import sys
import inspect

class MaximumNumberRetries(Exception):
    """Exception raised when maximum number of attempts to redesign operator has been reached

    Attributes:
        message -- Explanation of the error
    """

    def __init__(self, num_parents):
        super().__init__(f"Maximum Number of Retries for Operator: {num_parents} Parents")

class AdaptiveOperator():
    def __init__(self, client, sandbox, pset, toolbox, num_parents, num_offspring, max_num_retries=5, model="Qwen/Qwen3-Coder-Next-FP8"):
        self.llm_client = client
        self.llm_model = model

        #Problem Definition
        self.pset = pset
        self.toolbox = toolbox
        self.sandbox = sandbox

        #Number of times to retry generating LLM response 
        self.max_num_retries = max_num_retries 

        #Setup custom operator
        self.operator_design = None 
        self.current_operator_module = None
        self.operator_design_validated = False
        self.daytona_wrapper = None
        self.original_llm_prompt = None
        self.llm_prompt = None
        self.num_parents = num_parents
        self.num_offspring = num_offspring

        self.num_retries = 0

        #Enables us to import operator designs stored in temp folder 
        sys.path.append('/temp')
    
    #TODO: Define custom exception classes, and add to doc strings

    def reset_operator(self):
        self.operator_design = None 
        self.current_operator_module = None
        self.operator_design_validated = False
        self.llm_prompt = None

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

    def redesign_operator(self):
        #TODO: Create a counter of how many time it retries
        code = ""
        #Keeps generating until correct format is produced
        while self.num_retries <= self.max_num_retries:
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

            #Must contain the operator function
            if ("def crossover_individuals(" in code) or ("def mutate_individual(" in code):
                break

            self.num_retries += 1

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
        pickle_object(self.pset, "pset")

        #Uploads files to sandbox
        for i in range(self.num_parents):
            with open(f"temp/individual{i}.pkl", "rb") as f:
                content = f.read()
                self.sandbox.fs.upload_file(content, f"individual{i}.pkl")

        with open("temp/pset.pkl", "rb") as f:
            content = f.read()
            self.sandbox.fs.upload_file(content, "pset.pkl")
        
        #Attempts to execute - redesigns operator if fails
        while self.num_retries < self.max_num_retries: 
            print(f"Num retries: {self.num_retries}")
            #Inserts LLM-generated function into full operator code
            operator_code = textwrap.indent(self.operator_design, "    ")
            wrapper_text = self.daytona_wrapper.replace("INSERT_METHOD_DEFINITION_HERE", operator_code)

            try:
                compile(wrapper_text, "<sandbox>", "exec")
            
                #Execute the code in the sandbox
                response = self.sandbox.process.code_run(wrapper_text)

                #TODO: Add a timeout - give it 30 seconds to produce code

                #Must convert the pickled results back to desired form
                offspring = []
                for i in range(self.num_offspring):
                    offspring.append(unpickle_daytona_file(f"offspring{i}", self.sandbox)) 

                self.operator_design_validated = True

                return offspring
            
            #TODO: Better error handling
            except SyntaxError as e:
                print("Generated code has a syntax error:")
                print(e)
                print("Line:", e.lineno)
                print("Text:", e.text)

                #If an error occurs, attempt to redesign the LLM function
                self.redesign_operator()

            except Exception as e:
                error = self.sandbox.fs.download_file("error.txt")
                print("Error occured:")
                print(error.decode("utf-8")) #Prints error log

                #If an error occurs, attempt to redesign the LLM function
                self.redesign_operator()

        raise MaximumNumberRetries(self.num_parents)
    
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
            with open("temp/operator_design.py", "w") as f:
                #Ensures imports are present
                f.write("from deap import base, creator, tools, gp\n")
                f.write("import random\n")
                f.write(self.operator_design)

            self.current_operator_module = load_module("llm_operator", "temp/operator_design.py")

        #Attempt to apply operator locally
        if self.current_operator_module != None:
            try:
                offspring = self.apply_operator(individuals)

                #Ensure correct types
                for i in range(len(offspring)):
                    if (not isinstance(offspring[i], gp.PrimitiveTree)) and (isinstance(offspring[i], creator.Individual)):
                        offspring[i] = gp.PrimitiveTree(list(offspring[i]))
                    if not isinstance(offspring[i], creator.Individual):
                        offspring[i] = creator.Individual(offspring[i])

                #Only once the module has been operated locally, do we accept the design
                print("Design accepted - resetting retries")
                self.num_retries = 0

                return offspring
            
            #Redesign if code is unable to execute locally
            except Exception as e:
                #Get new individuals (TODO)
                print(f"Can't execute operator (num_parents: {self.num_parents}) locally..")
                print(e)
                self.redesign_operator()

                #For now, returns original individuals
                return individuals
            
        else:
            raise Exception("No module loaded for custom operator") #TODO: Sort out error handling
