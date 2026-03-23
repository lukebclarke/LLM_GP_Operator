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
    except (ZeroDivisionError, ValueError):
        return 1

def square(number):
    try:
        return number**2
    except OverflowError:
        return 10**10

def cube(number):
    try:
        return number**3
    except OverflowError:
        return 10**10

def protectedRoot(number):
    try:
        return math.sqrt(number)
    except Exception:
        return 1

def protectedLog(number):
    try:
        return math.log(number)
    except Exception:
        return 1

def protectedExp(number):
    try:
        return math.exp(number)
    except OverflowError:
        return 10**10
    except Exception:
        return 1

