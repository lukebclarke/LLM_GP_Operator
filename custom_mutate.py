from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from daytona import Daytona
from daytona import CodeRunParams

import json
import random

from util import get_individual_from_string
from util import get_string_from_individual
from util import clean_llm_output

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
        #Temporary - redesign to identify when stagnating
        print("Hello?")
        if self.current_mutation == None:
            self.redesign_prompt(individual, llm_client, sandbox)
            print("Redesigned")

        str_individual = get_string_from_individual(individual)

        #TODO: way to make this cleaner?
        wrapper = f"""
{self.current_mutation}

ind = {str_individual}
result = mutate(ind)
print(result)
"""

        #TODO: Clean LLM code (remove '''python from start and end)
        print(wrapper)

        try:
            #Execute the code in the sandbox
            #response = sandbox.process.code_run(wrapper, params=CodeRunParams(argv=[str_individual]))
            response = sandbox.process.code_run(wrapper)
            
            if response.exit_code != 0:
                return f"Error: {response.exit_code} {response.result}"
            
            output_str = str(response.result)
            print(output_str)
            individual = get_individual_from_string(output_str, self.pset)
            print(individual)

            return get_individual_from_string(output_str, self.pset)
        finally:
            pass #TODO: Sort this out. Clean sandbox?
    
    def redesign_prompt(self, individual, llm_client, sandbox):
        prompt="""
        Write a Python function called 'mutate(individual)' that returns a mutated version of the individual. 
        We are using the DEAP library. Include all necessary imports (including DEAP)
        Assume individual is a mutable list. We will pass 
        You may define any parameters you need (e.g., mu, sigma, indpb) inside the mutate function.
        Choose reasonable values and document them in comments.
        Do not rely on external global variables.

        An example mutation is given below:
        
        mu = 0.0
        sigma = 1.0
        indpb = 0.1

        size = len(individual)
        if not isinstance(mu, Sequence):
            mu = repeat(mu, size)
        elif len(mu) < size:
            raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
        if not isinstance(sigma, Sequence):
            sigma = repeat(sigma, size)
        elif len(sigma) < size:
            raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

        for i, m, s in zip(range(size), mu, sigma):
            if random.random() < indpb:
                individual[i] += random.gauss(m, s)

        return individual
        
        The function should return the individual as a string. Do not return any other text or data.

        Return raw Python code only as text, do not wrap it in markdown code blocks or backticks. 
        """
        #TODO: Should we include pset?
        #TODO: Check the code runs (maybe use random example with pset?)

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
