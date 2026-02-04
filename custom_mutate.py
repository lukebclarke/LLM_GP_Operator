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

from util import get_individual_from_string
from util import get_string_from_individual
from util import clean_llm_output
from util import pickle_object
from util import unpickle_object
from util import tree_to_list
from util import list_to_tree
from util import unpickle_daytona_file

import textwrap 

class CustomMutate():
    def __init__(self, client, pset, toolbox, max_num_retries=5):
        self.client = client
        self.pset = pset
        self.toolbox = toolbox
        self.current_mutation = None #TODO: Track the design we are currently using, and mutate according to this design
        self.max_num_retries = max_num_retries #Number of times to retry generating LLM response 

        #Wrapper for code
        with open("docs/mutation_wrapper.txt", "r") as f:
            self.mutation_wrapper = f.read()

        #Gets LLM Prompt from file
        with open("docs/LLMPromptMutation.txt", "rb") as f:
            self.llm_prompt = f.read()
    
    def llm_custom_mutate(self, individual, llm_client, sandbox):
        #Pickles objects - enables transfer to sandbox environment
        pickle_object(individual, "individual")
        pickle_object(self.pset, "pset")

        #Uploads files to sandbox
        with open("temp/individual.pkl", "rb") as f:
            content = f.read()
            sandbox.fs.upload_file(content, "individual.pkl")

        with open("temp/pset.pkl", "rb") as f:
            content = f.read()
            sandbox.fs.upload_file(content, "pset.pkl")

        #Inserts LLM-generated function into full mutation code
        mutation_code = textwrap.indent(self.current_mutation, "    ")
        wrapper_text = self.mutation_wrapper.replace("INSERT_CURRENT_MUTATION_HERE", mutation_code)
        
        #Attempts to execute - redesigns mutation if fails
        for i in range(self.max_num_retries): 
            try:
                compile(wrapper_text, "<sandbox>", "exec")
            
                #Execute the code in the sandbox
                response = sandbox.process.code_run(wrapper_text)

                #TODO: Add a timeout - give it 30 seconds to produce code

                #Must convert the pickled object back desired form
                output = unpickle_daytona_file("result", sandbox)

                return output
            
            except SyntaxError as e:
                print("Generated code has a syntax error:")
                print(e)
                print("Line:", e.lineno)
                print("Text:", e.text)

                #If an error occurs, attempt to redesign the LLM function
                self.redesign_prompt(individual, llm_client, sandbox)

            except Exception as e:
                error = sandbox.fs.download_file("error.txt")
                print("Error occured:")
                print(error.decode("utf-8")) #Prints error log

                #If an error occurs, attempt to redesign the LLM function
                self.redesign_prompt(individual, llm_client, sandbox)

        raise Exception("Too many attempts to generate code - redesign the LLM prompt")
    
    def redesign_mutation(self, llm_client):
        print("Redesigning mutation operator...")

        #TODO: Create a counter of how many time it retries
        code = ""
        #Keeps generating until correct format is produced
        while True:
            response = llm_client.chat.completions.create(
                model="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
                temperature=0.95,
                messages=[
                {
                    "role": "user",
                    "content": self.llm_prompt
                }
                ]
            )
                
            code = response.choices[0].message.content

            #Must contain the function
            if "def mutate_individual(" in code:
                break

        #Saves the resulting function - can be reaccessed
        self.current_mutation = clean_llm_output(code)

    def mutate(self, individual, llm_client, sandbox):
        #By default, use uniform mutation
        if self.current_mutation == None:
            return gp.mutUniform(individual, expr=self.toolbox.expr_mut, pset=self.pset)
        else:
            return self.llm_custom_mutate(individual, llm_client, sandbox)
