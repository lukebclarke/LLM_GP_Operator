from adaptive_regressor import AdaptiveRegressor
from standard_regressor import StandardRegressor

from pmlb import fetch_data, dataset_names
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gzip

dataset_name = 'feynman_II_38_14'
X, Y = fetch_data(dataset_name, return_X_y=True)

ao_est = AdaptiveRegressor(
    pop_size=200,
    gens=40,
    max_time=8.0*60.0*60.0,
    cxpb=0.7,
    mutpb=0.1,
    k=3,
    functions=["+", "-", "*", "/", "sqrt", "sin", "cos", "log"],
    verbose=True
)

standard_est = StandardRegressor(
    pop_size=200,
    gens=40,
    max_time=8.0*60.0*60.0,
    cxpb=0.7,
    mutpb=0.1,
    k=3,
    functions=["+", "-", "*", "/", "sqrt", "sin", "cos", "log"],
    verbose=True
)

standard_est.fit(X, Y)
print(standard_est.hof_[0])