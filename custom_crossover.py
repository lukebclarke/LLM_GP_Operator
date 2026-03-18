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

class CustomCrossover():
    def __init__(self, client, pset, toolbox, max_num_retries=5):
        self.client = client
        self.pset = pset
        self.toolbox = toolbox

        self.current_crossover = None #TODO: Track the design we are currently using, and crossover according to this design
        self.current_crossover_module = None
        self.design_validated = False
        self.max_num_retries = max_num_retries #Number of times to retry generating LLM response 

        #Wrapper for code
        with open("docs/crossover_wrapper.txt", "r") as f:
            self.crossover_wrapper = f.read()

        #Gets LLM Prompt from file
        with open("docs/LLMPromptCrossover.txt", "r") as f:
            self.original_llm_prompt = f.read()
        self.llm_prompt = None

        #Enables us to import crossover design stored in temp folder 
        sys.path.append('/temp')
    
    #TODO: Define custom exception classes, and add to doc strings

    def reset_crossover(self):
        self.current_crossover = None 
        self.current_crossover_module = None
        self.design_validated = False
        self.llm_prompt = None

    def llm_custom_crossover_daytona(self, individual1, individual2, llm_client, sandbox):
        """Used to execute LLM-generated code using the Daytona sandbox. Verifies that the code is able to execute successfully, and that it is trusted.

        Args:
            individual1 (gp.Individual): The first individual to crossover
            individual2 (gp.Individual): The first individual to crossover
            llm_client (Together): The LLM client used to redesign the crossover operator
            sandbox (Daytona): The Daytona sandbox used to execute the LLM-generated crossover design

        Raises:
            Exception: TODO

        Returns:
            gp.Individual, gp.Individual: The offspring of the crossover
        """

        #Pickles objects - enables transfer to sandbox environment
        pickle_object(individual1, "individual1")
        pickle_object(individual2, "individual2")
        pickle_object(self.pset, "pset")

        #Uploads files to sandbox
        with open("temp/individual1.pkl", "rb") as f:
            content = f.read()
            sandbox.fs.upload_file(content, "individual1.pkl")

        with open("temp/individual2.pkl", "rb") as f:
            content = f.read()
            sandbox.fs.upload_file(content, "individual2.pkl")

        with open("temp/pset.pkl", "rb") as f:
            content = f.read()
            sandbox.fs.upload_file(content, "pset.pkl")
        
        #Attempts to execute - redesigns crossover if fails
        for i in range(self.max_num_retries): 
            #Inserts LLM-generated function into full operator code
            crossover_code = textwrap.indent(self.current_crossover, "    ")
            wrapper_text = self.crossover_wrapper.replace("INSERT_CURRENT_CROSSOVER_HERE", crossover_code)

            try:
                compile(wrapper_text, "<sandbox>", "exec")
            
                #Execute the code in the sandbox
                response = sandbox.process.code_run(wrapper_text)

                #TODO: Add a timeout - give it 30 seconds to produce code

                #Must convert the pickled results back to desired form
                offspring1 = unpickle_daytona_file("offspring1", sandbox)
                offspring2 = unpickle_daytona_file("offspring2", sandbox)

                self.design_validated = True

                return offspring1, offspring2
            
            except SyntaxError as e:
                print("Generated code has a syntax error:")
                print(e)
                print("Line:", e.lineno)
                print("Text:", e.text)

                #If an error occurs, attempt to redesign the LLM function
                self.redesign_crossover(llm_client)

            except Exception as e:
                error = sandbox.fs.download_file("error.txt")
                print("Error occured:")
                print(error.decode("utf-8")) #Prints error log

                #If an error occurs, attempt to redesign the LLM function
                self.redesign_crossover(llm_client)

        raise Exception("Too many attempts to generate code - redesign the LLM prompt")
    
    def llm_custom_crossover_locally(self, individual1, individual2, llm_client):
        """Used to execute LLM-generated code locally. Used for trusted code.

        Args:
            individual1 (gp.Individual): The first parent
            individual2 (gp.Individual): The second parent
            llm_client (Together): The LLM client used to redesign the crossover

        Raises:
            Exception: TODO

        Returns:
            gp.Individual, gp.Individual: The offspring of the crossover operation
        """
        #TODO: Wrap in try except - redesign if crashes
        #TODO: Do we need to pass LLM_Client?

        #Ensure that the Python design is saved locally
        if self.current_crossover_module == None:
            with open("temp/crossover_design.py", "w") as f:
                #Ensures imports are present
                f.write("from deap import base, creator, tools, gp\n")
                f.write("import random\n")
                f.write(self.current_crossover)
            self.current_crossover_module = load_module("llm_crossover", "temp/crossover_design.py")

        #Attempt to crossover locally
        if self.current_crossover_module != None:
            try:
                ind1, ind2 = self.current_crossover_module.crossover_individuals(individual1, individual2, self.pset)

                #Ensure type individual
                if not isinstance(ind1, creator.Individual):
                    ind1 = creator.Individual(ind1)
                if not isinstance(ind2, creator.Individual):
                    ind2 = creator.Individual(ind2)

                return ind1, ind2
            
            #Redesign if code is unable to execute locally
            except Exception as e:
                #Get new individuals (TODO)
                print("Can't execute locally..")
                self.redesign_crossover(llm_client)

                #For now, returns original individuals
                return individual1, individual2
            
        else:
            raise Exception("No module loaded for custom crossover") #TODO: Sort out error handling

    def update_llm_prompt(self, history):
        #LLMs can deal with JSON
        formatted_history = json.dumps(history, indent=2)
        self.llm_prompt = self.original_llm_prompt.replace("INSERT_LOGBOOK_HERE", str(formatted_history))

    def redesign_crossover(self, llm_client):
        #TODO: Create a counter of how many time it retries
        code = ""
        #Keeps generating until correct format is produced
        while True:
            #TODO: Pass model to function so we can use same model for mutation and crossover
            response = llm_client.chat.completions.create(
                model="Qwen/Qwen3-Coder-Next-FP8",
                temperature=0.95,
                messages=[
                {"role": "system", "content": "You provide Python code to be directly executed out-of-the-box. Return only raw Python code, do not include any additional text/explanations in your response."},
                {
                    "role": "user",
                    "content": self.llm_prompt
                }
                ],
            )
                
            code = response.choices[0].message.content
            print(code)

            #Must contain the function
            if "def crossover_individuals(" in code:
                break

        #Saves the resulting function - can be reaccessed
        self.current_crossover = clean_llm_output(code)
        self.design_validated = False #TODO: Merge these variables
        self.current_crossover_module = None

    def crossover(self, individual1, individual2, llm_client, sandbox):
        #By default, use one point crossover
        if self.current_crossover == None:
            ind1, ind2 = gp.cxOnePoint(individual1, individual2)
            return ind1, ind2
        #Validates the design by using Daytona to execute the code
        elif self.current_crossover != None and self.design_validated == False:
            return self.llm_custom_crossover_daytona(individual1, individual2, llm_client, sandbox)
        #If the design has already been validated, can execute locally
        elif self.current_crossover != None and self.design_validated == True:
            #Ensure design is saved locally
            return self.llm_custom_crossover_locally(individual1, individual2, llm_client)

