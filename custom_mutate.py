from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import random

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
        if random.random() < 0.05:
            #Custom LLM Mutation
            return self.selectMutation_LLM(individual, client, base_prompt)
        else:
            #Otherwise, just perform mutation as normal
            return gp.mutUniform(individual, expr=self.toolbox.expr_mut, pset=self.pset)
