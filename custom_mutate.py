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
    
    def llm_custom_mutate(self, individual, llm_client, sandbox, base_prompt):
        #TODO: Temporary - redesign to identify when stagnating
        print("Mutating")

        if self.current_mutation == None:
            self.redesign_prompt(individual, llm_client, sandbox)
            print("Redesigned")

        #Prints original individual - used for debugging purposes
        print(f"Individual:\n{individual}")
        ind_list = tree_to_list(individual)
        print(f"Original Individual: {ind_list}")

        #Pickles objects - enables transfer to sandbox environment
        pickle_object(individual, "individual")
        pickle_object(self.pset, "pset")

        #Uploads files to sandbox
        with open("individual.pkl", "rb") as f:
            content = f.read()
            sandbox.fs.upload_file(content, "individual.pkl")

        with open("pset.pkl", "rb") as f:
            content = f.read()
            sandbox.fs.upload_file(content, "pset.pkl")

        print("Files uploaded to sandbox...")

        #TODO: way to make this cleaner?
        wrapper=f"""
with open("error.txt", "w") as f:
    f.write("Running...")

try: 
{textwrap.indent(self.current_mutation, "    ")}

    import pickle
    from deap import base, creator, tools, gp
    import math
    import operator
    #import gp_primitives
    from functools import partial
    import random

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    with open("error.txt", "a") as f:
        f.write("Set up creator")
   
    #Load pickled objects
    with open("pset.pkl", "rb") as f:
        pset = pickle.load(f)
    
    with open("individual.pkl", "rb") as f:
        individual = pickle.load(f)
    
    with open("error.txt", "a") as f:
        f.write("Loaded pickles")

    #Run mutation
    result = mutate_individual(individual, pset)

    with open("error.txt", "a") as f:
        f.write("Result:")
        f.write(str(result))

    #Save result
    with open("result.pkl", "wb") as f:
        pickle.dump(result, f)

except Exception as e:
    import traceback
    with open("error.txt", "a") as f:
        f.write(traceback.format_exc())
    raise
"""

        #TODO: Clean LLM code (remove '''python from start and end)
        print(wrapper)

        #Redesignes a maximum of 10 times
        for i in range(5): 
            #Detects syntax errors
            try:
                compile(wrapper, "<sandbox>", "exec")
            except SyntaxError as e:
                print("Generated code has a syntax error:")
                print(e)
                print("Line:", e.lineno)
                print("Text:", e.text)
                raise

            try:
                #Execute the code in the sandbox
                #response = sandbox.process.code_run(wrapper, params=CodeRunParams(argv=[str_individual]))
                response = sandbox.process.code_run(wrapper)

                if hasattr(e, "exit_code"):
                    print(f"Exit code: {e.exit_code}")

                if hasattr(e, "stderr"):
                    print(f"Error output: {e.stderr}")
                #TODO: Add a timeout - give it 30 seconds to produce code

                output = unpickle_daytona_file("result", sandbox)
                print(f"Output string: {output}")
                print("unpickled")

                return individual
            except Exception as e:
                #Prints error
                try:
                    print("YO")
                    error = sandbox.fs.download_file("error.txt")
                    print("PYTHON TRACEBACK:")
                    print(error.decode("utf-8"))
                except Exception:
                    print("No error.txt written - failure occurred before try/except")

                self.redesign_prompt(individual, llm_client, sandbox)
            finally:
                pass #TODO: Sort this out. Clean sandbox?

        #TODO: Raising this in right place?
        raise Exception("Too many attempts to generate code - redesign the LLM prompt")
    
    def redesign_prompt(self, individual, llm_client, sandbox):
        #Gets LLM Prompt from file
        with open("LLMPromptMutation.txt", "rb") as f:
            prompt = f.read()

        code = ""

        while True:
            try:
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
            except ValueError:
                #TODO: WE can remove this - only used it when we got LLM to make a choice
                print("Value Error - retrying")
                pass

            if "def mutate_individual(" in code:
                break
            else:
                print("No valid function found - retrying")

        self.current_mutation = code
        self.current_mutation = clean_llm_output(code)

    def mutate(self, individual, client, base_prompt):
        """Mutates the specified individual 

        Args:
            individual (Individual): _description_
            client (Together): TogetherAI client instance
            base_prompt (string): Base prompt used to form part of the LLM input

        Returns:
            Tuple: A mutated tree (from the original individual)
        """
        #TODO: Identify a point at which evolution stagnates

        #For now, we will use custom mutate randomly
        self.createMutationPrompt(base_prompt, individual)
        if random.random() < 0.00:
            #Custom LLM Mutation
            return self.selectMutation_LLM(individual, client, base_prompt)
        else:
            #Otherwise, just perform mutation as normal
            return gp.mutUniform(individual, expr=self.toolbox.expr_mut, pset=self.pset)
