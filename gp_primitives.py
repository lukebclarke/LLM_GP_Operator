"""
Defines the set of primitives used for the problems.
"""
import operator
import math
import random
from functools import partial

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1
