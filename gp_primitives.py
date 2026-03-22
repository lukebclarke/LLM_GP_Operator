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

def square(number):
    return number**2

def cube(number):
    return number**3

def protectedRoot(number):
    try:
        return math.sqrt(number)
    except ValueError:
        return 1

def protectedLog(number):
    try:
        return math.log(number)
    except ValueError:
        return 1

