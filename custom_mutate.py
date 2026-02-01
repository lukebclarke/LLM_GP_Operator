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

class CustomMutate():
    def __init__(self, client, base_prompt, pset, toolbox):
        self.client = client
        self.base_prompt = base_prompt
        self.pset = pset
        self.toolbox = toolbox
        self.current_mutation = None #TODO: Track the design we are currently using, and mutate according to this design

    def createMutationPrompt(self, base_prompt, individual):
        """Creates a custom mutation prompt for a LLM that includes information about the current state of the algorithm

        Args:
            base_prompt (string): The basic contents of the LLM input - does not include any information about the individual
            individual (Individual): The individual used to generate the prompt

        Returns:
            string: The full prompt to feed to LLM
        """
        #TODO: Include fitness, avg fitness, etc.
        #TODO: Don't do this based on individual, do based on generation
        #TODO: More optimal way to count number of nodes?
        #TODO: Redesign this whole prompt
        nodes, edges, labels = gp.graph(individual)
        num_nodes = len(nodes)
        height = individual.height
        additional_info = f"The current GP individual:\n- Has {num_nodes} nodes\n- Has a depth of {height}\n"

        return additional_info + base_prompt
    
    def getLLMChoice(self, individual, client, base_prompt):
        """Determines how to mutate future solutions by prompting a LLM 

        Args:
            individual (Individual): The individual used to generate the prompt
            client (Together): The Together LLM Client reference
            base_prompt (string): The prompt used by the LLM to create new mutation strategy

        Returns:
            int: The choice of mutation method - number corresponds to type of strategy 
        """

        #Gets LLM Prompt
        full_prompt = self.createMutationPrompt(base_prompt, individual)

        #Repeats until the LLM makes a decision
        while True:
            try:
                response = client.chat.completions.create(
                    model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                    temperature=0.95,
                    messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                    ]
                )
                
                choice = int(response.choices[0].message.content)

                #TODO: Ensure LLM choice is valid (within options)

                return choice
            #Ensures that the choice is an integer
            except ValueError:
                print("Retry")
                pass

    def selectMutation_LLM(self, individual, client, base_prompt):
        """Mutation operator is designed based on output from LLM

        Args:
            individual (Individual): The individual to mutate
            client (Together): The TogetherAI client
            base_prompt (string): The basic prompt that forms part of the LLM input

        Raises:
            ValueError: If no valid choice/design is made

        Returns:
            Tuple: The tuple representing one tree, after the individual has been mutated appropiately
        """
        #TODO: Provide more information to the LLM so it can better select an individual
        choice = self.getLLMChoice(individual, client, base_prompt)

        if choice == 1:
            #Uniform Mutation
            return gp.mutUniform(individual, expr=self.toolbox.expr_mut, pset=self.pset)
        elif choice == 2:
            #Gaussian Mutation
            return gp.mutShrink(individual)
        elif choice == 3:
            return gp.mutNodeReplacement(individual, pset=self.pset)
        else:
            #TODO: Handle this error 
            print("Invalid choice")
            raise ValueError
        
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
        wrapper = f"""
{self.current_mutation}

import pickle
from deap import base, creator, tools, gp

# Load pickled objects
with open("individual.pkl", "rb") as f:
    individual = pickle.load(f)

with open("pset.pkl", "rb") as f:
    pset = pickle.load(f)

print(individual)
print("")
print(pset)
# Run mutation
result = mutate_individual(individual, pset)

# Save result
with open("result.pkl", "wb") as f:
    pickle.dump(result, f)
"""

        #TODO: Clean LLM code (remove '''python from start and end)
        print(wrapper)

        #Redesignes a maximum of 10 times
        for i in range(5): 
            try:
                #Execute the code in the sandbox
                #response = sandbox.process.code_run(wrapper, params=CodeRunParams(argv=[str_individual]))
                response = sandbox.process.code_run(wrapper)
                #TODO: Add a timeout - give it 30 seconds to produce code
                
                if response.exit_code != 0:
                    raise Exception(f"Error: {response.exit_code} {response.result}")
                
                #output = response.result
                output = unpickle_daytona_file("result.pkl", sandbox)
                print(f"Output string: {output}")

                return individual
            except:
                print("Code failed to execute - redesigning prompt")
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
                            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
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
                pass

            if "def mutate" in code:
                break

        self.current_mutation = code
        self.current_mutation = clean_llm_output(code)

        print(self.current_mutation)

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
