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
    def __init__(self, client, base_prompt, pset, toolbox):
        self.client = client
        self.base_prompt = base_prompt
        self.pset = pset
        self.toolbox = toolbox
        self.current_mutation = None #TODO: Track the design we are currently using, and mutate according to this design

        #Wrapper for ::< Code
        with open("docs/mutation_wrapper.txt", "r") as f:
            self.mutation_wrapper = f.read()
    
    def llm_custom_mutate(self, individual, llm_client, sandbox, base_prompt):
        #TODO: Temporary - redesign to identify when stagnating. We should start with uniform mutation, instead of redesigning from the start.
        if self.current_mutation == None:
            self.redesign_prompt(individual, llm_client, sandbox)

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
        print(wrapper_text)
        
        #Will attempt to redesign a limited number of times
        for i in range(5): 
            try:
                compile(wrapper_text, "<sandbox>", "exec")
            
                #Execute the code in the sandbox
                response = sandbox.process.code_run(wrapper_text)

                print("Successfully executed")
                #TODO: Add a timeout - give it 30 seconds to produce code

                #Must convert the pickled object back to gp.Individual form
                output = unpickle_daytona_file("result", sandbox)
                print(f"Output string: {output}")

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
    
    def redesign_prompt(self, individual, llm_client, sandbox):
        print("Redesigning mutation operator...")

        #Gets LLM Prompt from file
        with open("docs/LLMPromptMutation.txt", "rb") as f:
            prompt = f.read()

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
                    "content": prompt
                }
                ]
            )
                
            code = response.choices[0].message.content

            #Must contain the function
            if "def mutate_individual(" in code:
                break

        #Saves the resulting function - can be reaccessed
        self.current_mutation = clean_llm_output(code)