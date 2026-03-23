import numpy as np
import operator
import math
from functools import partial
import random

#Sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, RegressorMixin, _fit_context
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

#DEAP
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

#Custom Classes
import gp_primitives
from evolutionary_algorithm import DynamicOperators

class AdaptiveRegressor(BaseEstimator, RegressorMixin):
    """A scikit-learn regressor model for an evolutionary algorithm with adaptive operators

    Parameters
    ----------
    pop_size : int, default=200
        Size of evolutionary population

    gens : int, default=40
        Number of generations

    max_time : float, default=8*60*60
        Maximum time permitted before execution times out

    cxpb : float, default=0.6
        Probability of crossover

    cxpb : float, default=0.1
        Probability of mutation

    k : int, default=3
        Number of consecutive generations to required with no fitness improvement to redesign genetic operators

    functions : list, default=['+','-','*','/','^2','^3','sqrt','sin','cos','exp','log']
        List of functions available within evolved genetic trees

    verbose : bool, default=True
        Dictates whether to print log during fitting


    Attributes
    ----------
    is_fitted_ : bool
        A boolean indicating whether the estimator has been fitted.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        #TODO: Add feature names?

    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "pop_size": [int],
        "gens": [int],
        "max_time": [float],
        "cxpb": [float],
        "mutpb": [float],
        "k": [int],
        "functions": [list],
        "verbose": [bool]
    }

    def __init__(self, pop_size=200, gens=40, max_time=8.0*60.0*60.0, cxpb=0.6, mutpb=0.1, functions=['+','-','*','/','^2','^3','sqrt','sin','cos','exp','log'], k=3, verbose=True):
        self.pop_size = pop_size
        self.gens = gens
        self.max_time = max_time
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.functions = functions
        self.k = k
        self.verbose = verbose

        #Variables accessed after fitting
        self.final_pop_ = None
        self.hof_ = None
        self.stats_ = None
        self.logbook_ = None

    def create_pset(self, n_features):
        pset = gp.PrimitiveSet("MAIN", arity=n_features) #Program takes one input

        #Describes what each function corresponds to
        operator_map = {
            "+": (operator.add, 2),
            "-": (operator.sub, 2),
            "*": (operator.mul, 2),
            "/": (gp_primitives.protectedDiv, 2),
            "^2": (gp_primitives.square, 1),
            "^3": (gp_primitives.cube, 1),
            "sqrt": (gp_primitives.protectedRoot, 1),
            "sin": (math.sin, 1),
            "cos": (math.cos, 1),
            "exp": (gp_primitives.protectedExp, 1),
            "log": (gp_primitives.protectedLog, 1)
        }

        #Creates a DEAP Primitive Set containing each function
        for f in self.functions:
            func_details = operator_map[f]
            pset.addPrimitive(func_details[0], func_details[1])

        #Adds ephemeral constants
        #TODO: Do we define what values they can take in problem definition?
        pset.addEphemeralConstant("rand101", partial(random.randint, -1, 1)) #Program can create random constants between 0 and 1

        return pset

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Evolves an evolutionary model with LLM-based adaptive operators. Identifies the best solution.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers).

        Returns
        -------
        self : object
            Returns self.
        """
        #TODO: Check this line
        # X, y = self._validate_params(X, y, accept_sparse=True)

        #Remove headers, if applicable
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values
        n_features = X.shape[1]

        #Gets problem set
        pset = self.create_pset(n_features)

        #Runs evolutionary algorithm with pset
        algorithms = DynamicOperators(self.pop_size, pset, X, y, self.k)
        # self.final_pop, self.logbook, self.hof, self.stats = algorithms.runDynamicEA(self.cxpb, self.mutpb, self.gens, verbose=self.verbose)
        self.final_pop, self.logbook, self.hof = algorithms.runSimpleEA()

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Finds the value for a set of input variables, X, using our best found solution from the evolutionary process.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of values calculated from the best evolved solution .
        """
        if check_is_fitted(self):
            #Finds best solution, and compiles it into an equation
            best_solution = self.hof[0]
            func = self.toolbox.compile(expr=best_solution) 

        # X = self._validate_params(X)

        return np.array([func(*row) for row in X])
