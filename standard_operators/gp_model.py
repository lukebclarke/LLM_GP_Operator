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

class StandardRegressor(BaseEstimator, RegressorMixin):
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
        "verbose": [bool],
        "random_state": [int]
    }

    def __init__(self, pop_size=200, gens=40, max_time=8.0*60.0*60.0, cxpb=0.6, mutpb=0.1, functions=['+','-','*','/','^2','^3','sqrt','sin','cos','exp','log'], verbose=True, random_state=None):
        self.pop_size = pop_size
        self.gens = gens
        self.max_time = max_time
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.functions = functions
        self.verbose = verbose
        self.random_state = random_state

        #Seeds run
        if not self.random_state:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        #Variables accessed after fitting
        self.final_pop_ = None
        self.hof_ = None
        self.stats_ = None
        self.logbook_ = None
        self.toolbox_ = None

    def create_pset(self, n_features):
        """Defines the primitive set based on the defined functions and number of features

        Args:
            n_features (int): The number of features of X

        Returns:
            gp.PrimitiveSet: The custom primitive set
        """
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
    
    def create_toolbox(self, pset, X, Y):
        """Creates toolbox which contains genetic operator

        Args:
            pset (gp.PrimitiveSet): The primitive set to use
            X {array-like, sparse matrix}, shape (n_samples, n_features): The training input samples.
            Y {array-like}, shape (n_samples,) or (n_samples, n_outputs): The target values

        Returns:
            base.Toolbox: The DEAP toolbox object for problem
        """
        # Defines 'toolbox' functions we can use to create and evaluate individuals
        toolbox = base.Toolbox() 
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2) #Generates random expressions (some full trees, other small ones)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr) #Creates individuals
        toolbox.register("population", tools.initRepeat, list, toolbox.individual) #Creates populations
        toolbox.register("compile", gp.compile, pset=pset) #Converts tree into runnable code 
        toolbox.register("evaluate", self.evaluateIndividual, toolbox, X, Y)
        toolbox.register("select", tools.selTournament, tournsize=3) #TODO: Add tournament size to hyper-parameters
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        #Defines limits for genetic operations
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)) #Limits height of tree
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        return toolbox
    
    def evaluateIndividual(self, toolbox, X, Y, individual):
        """Calculates mean squared error (MSE) for individual/program (i.e. calculates fitness)

        Args:
            individual (gp.Individual): The individual to evaluate

        Returns:
            float: The MSE value
        """
        #TODO: Work for multiple Y values
        #Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual) 

        #Evaluate the mean squared error between the expression and the real function
        sqerrors = np.array([(func(*x) - y)**2 for x, y in zip(X, Y)])
        sqerrors = sqerrors.flatten()

        return math.fsum(sqerrors) / len(X),

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Evolves an evolutionary model with standard genetic operators. Identifies the best solution.

        Args:
            X {array-like, sparse matrix}, shape (n_samples, n_features): The training input samples.
            y array-like, shape (n_samples,) or (n_samples, n_outputs): The target values (real numbers).

        Returns:
            StandardRegressor: Returns self.
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

        self.toolbox_ = self.create_toolbox(pset, X, y)

        self.final_pop_ = self.toolbox_.population(n=self.pop_size)
        self.hof_ = tools.HallOfFame(1) #We track 1 best solution

        #Track statistics
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        self.mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        self.mstats.register("avg", np.mean)
        self.mstats.register("std", np.std)
        self.mstats.register("min", np.min)
        self.mstats.register("max", np.max)
        self.fitness_improvements = []
        self.redesign_generations = []
        self.stats_ = {"fitness_improvements": None}

        #Run simple EA
        self.final_pop_, self.logbook_ = algorithms.eaSimple(self.final_pop_, self.toolbox_, self.cxpb, self.mutpb, self.gens, stats=self.mstats,
                                    halloffame=self.hof_, verbose=True)
        
        #Finds minimum fitness per generation
        fitness_per_gens = self.logbook_.chapters["fitness"].select("min")
        fitness_improvements = [np.nan]
        for i in range(1, len(fitness_per_gens)):
            prev_gen_fitness = fitness_per_gens[i-1]
            current_gen_fitness = fitness_per_gens[i]
            fitness_improvements.append(prev_gen_fitness - current_gen_fitness)
        self.stats_["fitness_improvements"] = fitness_improvements

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Finds the value for a set of input variables, X, using our best found solution from the evolutionary process.

        Args:
            X {array-like, sparse matrix}, shape (n_samples, n_features): The training input samples

        Returns:
            ndarray, shape (n_samples,): Returns an array of values calculated from the best evolved solution
        """
        if self.is_fitted_:
            #Finds best solution, and compiles it into an equation
            best_solution = self.hof_[0]
            func = self.toolbox_.compile(expr=best_solution) 

        #TODO: Check line
        # X = self._validate_params(X)

        return np.array([func(*row) for row in X])
