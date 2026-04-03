import numpy as np
import operator
import math
from functools import partial
import random
import os
import time
import threading

#API Keys
from dotenv import load_dotenv

#Clients
from together import Together
from daytona import Daytona

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
from adaptive_operators.adaptive_gp import AdaptiveGP
from adaptive_operators.custom_crossover import CustomCrossover
from adaptive_operators.custom_mutation import CustomMutation
from util import pickle_object

class AdaptiveRegressor(BaseEstimator, RegressorMixin):
    """A scikit-learn regressor model for an evolutionary algorithm with adaptive operators
    TODO - Update doc string

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

    def __init__(self, pop_size=200, gens=40, max_time=8.0*60.0*60.0, cxpb=0.6, mutpb=0.1, functions=['+','-','*','/','^2','^3','sqrt','sin','cos','exp','log'], k=3, verbose=True, timeout=20, random_state=None):
        self.pop_size = pop_size
        self.gens = gens
        self.max_time = max_time
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.functions = functions
        self.k = k
        self.verbose = verbose
        self.timeout = timeout
        self.random_state = random_state

        #Seeds run
        if not self.random_state:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        #Setup LLM client
        load_dotenv()
        self.client = self.setup_llm()
        self.pset = None
        self.sandbox = None
        self.toolbox = None
        self.custom_mutate = None
        self.custom_crossover = None

        #Variables accessed after fitting
        self.final_pop_ = None
        self.hof_ = None
        self.stats_ = None
        self.logbook_ = None

        self.algorithms_ = None
    
    def setup_llm(self):
        """Sets up Together LLM client

        Raises:
            Exception: API Key not found in .env file

        Returns:
            Together: The Together LLM client
        """
        #Defines LLM Client for custom genetic operators
        api_key = os.environ.get("TOGETHER_AI") #Uses the TogetherAI API

        if api_key is None:
            raise Exception("Together API key not found")

        client = Together(api_key=api_key)

        return client
    
    def initialise_sandbox_instance(self, result):
        """Sets up sandbox - to be used in threading context

        Args:
            result (dict): The dictionary used to store results. 'sandbox' provides access to the Daytona sandbox.
            timeout (float): The number 
        """
        #Create Daytona sandbox client
        daytonaClient = Daytona()
        sandbox = daytonaClient.create(timeout=self.timeout)
        result["sandbox"] = sandbox
        print("Sandbox setup")
        
        #Install DEAP
        sandbox.process.exec("python -m pip install deap==1.4.1", timeout=self.timeout)
        print("DEAP installed")
        result["sandbox"] = sandbox

        #Provide functions for pset
        with open("gp_primitives.py", "rb") as f:
            content = f.read()
            sandbox.fs.upload_file(content, "gp_primitives.py", timeout=self.timeout)

        #Upload Pset
        pickle_object(self.pset, "pset")
        with open("temp/pset.pkl", "rb") as f:
            content = f.read()
            sandbox.fs.upload_file(content, "pset.pkl")

        result["sandbox"] = sandbox
    
    def setup_daytona(self, max_attempts=10):
        """Attempts to setup Daytona sandbox, retrying if it fails initialisation.

        Args:
            max_attempts (int, optional): The number of attempts to retry before returning RuntimeError. Defaults to 10.

        Raises:
            RuntimeError: Raised if sandbox is reinitialised too many times

        Returns:
            Daytona: Daytona sandbox client
        """
        for i in range(max_attempts):
            result = {"sandbox": None}
            try:
                result["sandbox"] = None

                #Uses threads to implement timeout
                t = threading.Thread(target=self.initialise_sandbox_instance, args=(result,))
                t.start()
                t.join(self.timeout)

                if t.is_alive() or result["sandbox"] == None:
                    raise TimeoutError("Operation timed out")

                self.sandbox = result["sandbox"]

                return self.sandbox
            
            except TimeoutError:
                print("Sandbox initialisation failed...")

                #Attempt to delete sandbox
                try:
                    result["sandbox"].delete()
                except Exception:
                    #Add delay before retrying
                    time.sleep(1)

        raise RuntimeError("Too many attempts to initialise Daytona sandbox")
    
    def shutdown_sandbox(self):
        """
        Deletes/shutdowns the sandbox
        """
        self.sandbox.delete()

    def create_pset(self, n_features):
        """Defines the primitive set based on the defined functions and number of features

        Args:
            n_features (int): The number of features of X

        Returns:
            gp.PrimitiveSet: The custom primitive set
        """
        #Defines the number of inputs for problem as the number of features
        pset = gp.PrimitiveSet("MAIN", arity=n_features) 

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
    
    def create_toolbox(self, X, y):
        """Creates toolbox which contains genetic operators """
        #Defines 'toolbox' functions we can use to create and evaluate individuals
        self.toolbox = base.Toolbox() 
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2) #Generates random expressions (some full trees, other small ones)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr) #Creates individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual) #Creates populations
        self.toolbox.register("compile", gp.compile, pset=self.pset) #Converts tree into runnable code 
        self.toolbox.register("evaluate", self.evaluate_individual, X=X, Y=y)
        self.toolbox.register("select", tools.selTournament, tournsize=3) #TODO: Add tournament size to hyper-parameters
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)

        #Defines custom mutation + crossover interfaces
        self.custom_mutate = CustomMutation(self.client, self.sandbox, self.pset, self.toolbox, model="Qwen/Qwen3-Coder-Next-FP8", max_num_retries=15, max_local_skips=(0.1*self.pop_size))
        self.custom_crossover = CustomCrossover(self.client, self.sandbox, self.pset, self.toolbox, model="Qwen/Qwen3-Coder-Next-FP8", max_num_retries=15, max_local_skips=(0.1*self.pop_size))

        #Registers custom mutation + crossover methods
        self.toolbox.register("mate", self.custom_crossover.crossover)
        self.toolbox.register("mutate", self.custom_mutate.mutate) 

        #Defines limits for genetic operations
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)) #Limits height of tree
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    def evaluate_individual(self, individual, X, Y):
        """Calculates mean squared error (MSE) for individual/program (i.e. calculates fitness)

        Args:
            individual (gp.Individual): The individual to evaluate

        Returns:
            float: The MSE value
        """
        #TODO: Work for multiple Y values
        #Transform the tree expression in a callable function
        func = self.toolbox.compile(expr=individual) 

        #Evaluate the mean squared error between the expression and the real function
        sqerrors = np.array([(func(*x) - y)**2 for x, y in zip(X, Y)])
        sqerrors = sqerrors.flatten()

        return math.fsum(sqerrors) / len(X),


    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Evolves an evolutionary model with LLM-based adaptive operators. Identifies the best solution.

        Args:
            X {array-like, sparse matrix}, shape (n_samples, n_features): The training input samples.
            y array-like, shape (n_samples,) or (n_samples, n_outputs): The target values (real numbers).

        Returns:
            AdaptiveRegressor: Returns self.
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
        if not self.pset:
            self.pset = self.create_pset(n_features)

        #Initialises Daytona client
        if not self.sandbox:
            self.setup_daytona(max_attempts=10)

        self.create_toolbox(X, y)

        #If we have already initialised regressor, don't load up operators class again (sandbox takes long time to initialise)
        if self.algorithms_ == None:
            self.algorithms_ = AdaptiveGP(self.pop_size, self.pset, self.toolbox, self.client, self.sandbox, self.custom_mutate, self.custom_crossover, X, y, self.k)
        #If we have already run algorithm, reset all variables
        else:
            self.final_pop_ = None
            self.hof_ = None
            self.stats_ = None
            self.logbook_ = None

        self.final_pop_, self.logbook_, self.hof_, self.stats_ = self.algorithms_.run_adaptive_ea(self.cxpb, self.mutpb, self.gens, verbose=self.verbose)

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
            func = self.toolbox.compile(expr=best_solution) 

        #TODO: Check line
        # X = self._validate_params(X)
        print(func)
        return np.array([func(*row) for row in X])
