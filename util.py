from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

def get_individual_from_string(individual_string, pset):
    individual = gp.PrimitiveTree.from_string(individual_string, pset)

    return individual

def get_string_from_individual(individual_obj):
    ind_str = str(individual_obj)

    return ind_str

def clean_llm_output(output):
    output = output.strip()
    output = output.replace("```", "")
    output = output.replace("python", "")
    
    return output

#TODO: Remove this - used for testing
output = """
```python
import random
from deap import tools
from itertools import repeat
from typing import Sequence

def mutate(individual):
    # Mutation parameters
    mu = 0.0  # Mean of the Gaussian distribution
    sigma = 1.0  # Standard deviation of the Gaussian distribution
    indpb = 0.1  # Probability of an individual to be mutated

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

    return ''.join(map(str, individual))
```

ind = cos(-1)
result = mutate(ind)
print(result)"
"""

print(clean_llm_output(output))