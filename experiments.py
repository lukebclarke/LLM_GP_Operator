from adaptive_operators.gp_model import AdaptiveRegressor
from standard_operators.gp_model import StandardRegressor
from adaptive_operators.base_operator import MaximumNumberRetries
from util import get_similarity

from pmlb import fetch_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
import random
import math
import shutil
import time
from itertools import product

import scipy.stats as stats
import statistics

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from sklearn.model_selection import train_test_split

from matplotlib.colors import TABLEAU_COLORS, same_color

def pad_runs_multiple_algorithms(algorithm_runs):
    """Pads runs from multiple algorithms

    Args:
        algorithm_runs ([[[float]]]): An array containing a list of each algorithms, and the runs within that algorithm

    Returns:
        list: The padded list of algorithm runs
    """
    algorithm_counts = [len(alg) for alg in algorithm_runs]

    #Flattens
    flattened_runs = [run for alg in algorithm_runs for run in alg]

    #Pads runs
    padded_runs = pad_runs_single_algorithm(flattened_runs)

    #Returns to original shape
    separated_runs = []
    idx = 0
    for count in algorithm_counts:
        separated_runs.append(padded_runs[idx:idx+count])
        idx += count

    return separated_runs

def pad_runs_single_algorithm(runs):
    """Pads runs to a specified length for easy comparison

    Args:
        runs ([list]): A list of runs

    Returns:
        [list]: A list of padded runs
    """
    padded_runs = []

    #Finds the maximum length of a run
    max_len = max([len(run) for run in runs])

    #Pads all runs to the maximum length
    for run in runs:
        run = run + ([0] * ((max_len) - len(run)))
        padded_runs.append(run)

    return padded_runs

def analyse_across_runs(minimum_fitnesses):
    """Finds median, lower quartile and upper quartile across multiple runs

    Args:
        minimum_fitnesses (list): All minimum testing fitnesses

    Returns:
        list, list, list: A list of the medians, lower quartiles and upper quartiles across runs
    """
    #Runs must be padded already
    minimum_fitnesses = np.array(minimum_fitnesses)

    medians = np.median(minimum_fitnesses, axis=0)
    q1s = np.percentile(minimum_fitnesses, 25, axis=0)
    q3s = np.percentile(minimum_fitnesses, 75, axis=0)

    return medians, q1s, q3s

def plot_minimum_fitnesses(alg_names, alg_min_fitnesses, filepath=None):
    """Plots a graph of minimum fitness per generation

    Args:
        alg_names ([str]): The names of the models
        alg_min_fitnesses ([float]): The list of fitnesses
        filepath (str, optional): The location to save the plot. Defaults to None.
    """
    #Pad the minimum fitnesses (so each run is the same length)
    runs = np.array(pad_runs_multiple_algorithms(alg_min_fitnesses))

    fig = plt.figure(figsize=[7,5])
    ax = plt.subplot(111)
    for i in range(len(runs)):

        median, q1, q3 = analyse_across_runs(runs[i])
        print(median)
        x = np.arange(len(median))
        ax.plot(x, median, label=alg_names[i])
        ax.fill_between(x, q1, q3, alpha=0.2)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Minimum Fitness")
        ax.set_title("Minimum Fitness per Generation")
        ax.legend()

    if filepath:
        plt.savefig(filepath, dpi=300)
        plt.close()

def plot_avg_size(alg_names, alg_avg_sizes, filepath=None):
    """Plots a graph of average size per generation

    Args:
        alg_names ([str]): The names of the models
        alg_avg_sizes ([float]): The list of average sizes
        filepath (str, optional): The location to save the plot. Defaults to None.
    """
    #Pad the minimum fitnesses (so each run is the same length)
    runs = np.array(pad_runs_multiple_algorithms(alg_avg_sizes))

    fig = plt.figure(figsize=[7,5])
    ax = plt.subplot(111)

    #Analyses each run
    for i in range(len(runs)):
        median, q1, q3 = analyse_across_runs(runs[i])
        x = np.arange(len(median))
        ax.plot(x, median, label=alg_names[i])
        ax.fill_between(x, q1, q3, alpha=0.2)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Average Size")
        ax.set_title("Average Size per Generation")
        ax.legend()

    if filepath:
        plt.savefig(filepath, dpi=300)
        plt.close()

def plot_histogram_redesign_gens(redesign_gens, n_gens=100, filepath=None):
    """Plots a histogram for generations with most frequent operator redesigns

    Args:
        redesign_gens ([int]): Lists of generations that operators are redesigned on
        n_gens (int, optional): The number of generations. Defaults to 100.
        filepath (str, optional): The location to save the histogram. Defaults to None.
    """
    array = np.array(redesign_gens)
    flattened_gens = array.flatten()

    fig = plt.figure(figsize=[7,5])
    ax = plt.subplot(111)

    x, y = np.unique_counts(flattened_gens)

    hist = ax.hist(x, y, bins=n_gens)
    ax.set_title("Frequency of Redesigns per Generation")

    if filepath:
        plt.savefig(filepath, dpi=300)
        plt.close()

def plot_boxplot(names, values, title, y_label, filepath=None):
    """Plots a boxplot

    Args:
        names ([str]): Names of models
        values ([[float]]): The values for each model
        title (str): Title of the plot
        y_label (str): Label for the y-axis
        filepath (str, optional): The location to save the plot. Defaults to None.
    """
    fig = plt.figure(figsize=[7,5])
    ax = plt.subplot(111)

    #Generates box plot
    bplot = ax.boxplot(values, tick_labels=names, patch_artist=True, showfliers=False)

    #Gets default matplotlib colours
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    #Applies colours to boxplots
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.2)

    #Applies colours to median lines
    for median, color in zip(bplot['medians'], colors):
        median.set_color(color)

    ax.set_title(title)
    ax.set_ylabel(y_label)

    if filepath:
        plt.savefig(filepath, dpi=300)
        plt.close()

def bar_chart(graph_title, metric_name, bar_labels, values, filepath=None):
    """Plots bar chart

    Args:
        graph_title (str): Title of the graph
        metric_name (str): Name of the metric
        bar_labels ([str]): List of labels for each bar
        values ([float]): List of values to plot
        filepath (str, optional): The location to save the plot. Defaults to None.
    """
    fig, ax = plt.subplots()

    #Gets default matplotlib colours
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    #Generates bar chart
    bars = ax.bar(bar_labels, values, label=bar_labels, facecolor=colors)

    #Lowers transparency of colours
    for bar in bars:
        bar.set_alpha(0.2)

    ax.set_ylabel(metric_name)
    ax.set_title(graph_title)

    if filepath:
        plt.savefig(filepath, dpi=300)
    plt.close()

def similarity_bar_chart(model_names, mutation_similarity, crossover_similarity, filepath=None):
    """Plots the difference in mutation and crossover similarity for multiple models

    Args:
        model_names ([str]): Names of the models
        mutation_similarity ([float]): The mutation similarity for each model
        crossover_similarity ([float]): The mutation similarity for each model
        filepath (str, optional): The location to save the file. Defaults to None.
    """
    fig, ax = plt.subplots()

    w, x = 0.4, np.arange(len(model_names))

    #Generates bar chart
    mut_bar = ax.bar(x - w/2, mutation_similarity, w, label='Mutation')
    cross_bar = ax.bar(x + w/2, crossover_similarity, w, label='Crossover')
    #Lowers transparency of colours
    for bar in mut_bar:
        bar.set_alpha(0.2)
    for bar in cross_bar:
        bar.set_alpha(0.2)

    ax.set_xticks(x, model_names)
    ax.set_ylabel('Similarity (%)')
    ax.set_title('Operator Similarity')
    ax.legend()

    if filepath:
        plt.savefig(filepath, dpi=300)
        plt.close()

def scatter_plot(graph_title, metric1, metric2, model_names, xs, ys, filepath=None):
    """Plots a scatter graph between two metrics

    Args:
        graph_title (str): Name of the graph
        metric1 (str): Name of the first metric
        metric2 (str): Name of the second metric
        model_names ([str]): Names of the models
        xs ([float]): The values of metric 1
        ys ([float]): The values of metric 2
        filepath (str, optional): The location to save the graph. Defaults to None.
    """
    fig = plt.figure(figsize=[7,5])
    ax = plt.subplot(111)

    #Gets default matplotlib colours
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        ax.scatter(x, y, label=model_names[i], color=colors[i])

    ax.set_title(graph_title)
    ax.set_xlabel(metric1)
    ax.set_ylabel(metric2)
    ax.legend()

    if filepath:
        plt.savefig(filepath, dpi=300)
        plt.close()

def statistical_testing(alg1_fitnesses, alg2_fitnesses, alpha):
    """Performs Wilcoxon Signed Rank Test

    Args:
        alg1_fitnesses ([float]): Fitnesses to be compared from algorithm 1
        alg2_fitnesses ([float]): Fitnesses to be compared from algorithm 1
        alpha (float): The significance value

    Returns:
        float, float, bool: The p-value, the test statistic, and a boolean representing whether there is a significant difference between the algorithms
    """
    stat, p_value = stats.wilcoxon(alg1_fitnesses, alg2_fitnesses)

    print("Wilcoxon Signed Rank Test Statistic:", stat)
    print("p-value:", p_value)

    diff = False
    if p_value < alpha:
        diff = True

    return stat, p_value, diff

def compare_problem_types(ground_truth_wins, ground_truth_losses, ground_truth_ties, black_box_wins, black_box_losses, black_box_ties, ground_truth_improvements, black_box_improvements):
    """Creates a bar chart and boxplot comparing how LLM-GP performs compared to against standard GP on different problem types.

    Args:
        ground_truth_wins (int): Number of LLM-GP wins against standard GP on ground truth problems
        ground_truth_losses (int): Number of LLM-GP losses against standard GP on ground truth problems
        ground_truth_ties (int): Number of LLM-GP ties against standard GP on ground truth problems
        black_box_wins (int): Number of LLM-GP wins against standard GP on black box problems
        black_box_losses (int): Number of LLM-GP losses against standard GP on black box problems
        black_box_ties (int): Number of LLM-GP ties against standard GP on black box problems
        ground_truth_improvements ([float]): Percent improvement on ground truth problems of LLM-GP over standard GP
        black_box_improvements ([float]): Percent improvement on black box problems of LLM-GP over standard GP
    """
    x = ["Ground Truth", "Black Box"]
    wins = np.array([ground_truth_wins, black_box_wins])
    losses = np.array([ground_truth_losses, black_box_losses])
    ties = np.array([ground_truth_ties, black_box_ties])

    #Plot stacked bar chart
    plt.bar(x, wins, color='tab:green', alpha=0.2)
    plt.bar(x, ties, bottom=wins, color='tab:gray', alpha=0.2)
    plt.bar(x, losses, bottom=wins+ties, color='tab:red', alpha=0.2)
    plt.ylabel("Number of Problems")
    plt.legend(["Win", "Tie", "Losses"])
    plt.title("Performance of LLM-Based GP against Standard GP")
    plt.savefig("results/groundtruth_blackbox.pdf", dpi=300)
    plt.show()

    #Boxpot for Improvements
    plot_boxplot(x, [ground_truth_improvements, black_box_improvements], "LLM-Based GP Improvement", "Improvement over Standard GP (%)", "results/improvement_problem.pdf")

def run_problem_instance(problem_name, params, model_name, num_runs=10, save_results=True):
    """Runs a problem instance multiple times, collating the results

    Args:
        problem_name (str): Name of the problem from the PMLB dataset
        params (dict): The model parameters
        model_name (str): The name of the model
        num_runs (int, optional): Then number of iterations of each problem. Defaults to 10.
        save_results (bool, optional): Specifies whether to save the results. Defaults to True.

    Returns:
        dict: Contains all collated results from the runs
    """
    #Gets training and testing data
    print(problem_name)
    X, Y = fetch_data(problem_name, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #Make for model within results
    if save_results:
        directory_name = f"results/{problem_name}"
        directory_name = f"{directory_name}/{model_name}"
        try:
            os.mkdir(directory_name)
        except FileExistsError:
            pass

        #Opens log file
        log = open(f"{directory_name}/logbook.txt", "w")

    #Statistics
    all_size_avg = []
    all_fit_min = []
    all_fit_testing = []
    num_mutation_redesigns = []
    num_crossover_redesigns = []
    best_mutation_designs = []
    best_crossover_designs = []
    crossover_similarities = []
    mutation_similarities = []
    execution_times = []
    n_evals = []
    solved = []
    redesign_gens = []
    final_stats = {}

    for i in range(num_runs):
        
        #Deletes temp folder if already exists
        if os.path.exists("temp"):
            shutil.rmtree("temp")

        #Create fresh temp folder (and sub-folders)
        os.makedirs("temp")
        os.makedirs("temp/mutation_designs")
        os.makedirs("temp/crossover_designs")

        #Defines random seed
        params["random_state"] = 42 + i
        
        #TODO: Prevent this from crashing altogether
        for i in range(10):
            start_time = time.time()

            #Run adaptive evolutionary algorithm without re-initialising sanbdox
            try:
                ao_est = AdaptiveRegressor(**params)
                ao_est.fit(X_train, y_train)
                end_time = time.time()
                break
            except MaximumNumberRetries:
                continue            

        ao_est.shutdown_sandbox()

        #Get fitnesses on testing data
        ao_test = ao_est.predict(X_test)

        #Evaluate the mean squared error between the expression and the real function
        all_fit_testing.append(np.mean((ao_test - y_test) ** 2))

        #Finds average size and minimum fitness for adaptive EA
        all_size_avg.append(ao_est.logbook_.chapters["size"].select("avg"))
        all_fit_min.append(ao_est.logbook_.chapters["fitness"].select("min"))

        #Finds the number of operator redesigns (for mutation and crossover)
        num_mutation_redesigns.append(ao_est.stats_["num_mutation_redesigns"])
        num_crossover_redesigns.append(ao_est.stats_["num_crossover_redesigns"])

        #Finds the similarity between operator designs (for mutation and crossover)
        mutation_similarities.append(ao_est.stats_["mutation_similarity"])
        crossover_similarities.append(ao_est.stats_["crossover_similarity"])

        redesign_gens = ao_est.stats_["redesign_generations"]
        fitness_improvement_per_gen_ao = ao_est.stats_["fitness_improvements"]
        redesign_gens.append(redesign_gens)

        #Finds the best operator designs in the run (for mutation and crossover)
        best_mutation_design = ao_est.stats_["best_mutation_design"]
        best_crossover_design = ao_est.stats_["best_crossover_design"]

        if best_mutation_design:
            best_mutation_designs.append(best_mutation_design)

        if best_crossover_design:
            best_crossover_designs.append(best_crossover_design)

        #Time Elapsed
        elapsed_time = end_time - start_time
        execution_times.append(elapsed_time)

        #Number of evaluations
        n_evals.append(ao_est.stats_["n_evals"])

        #Determines if problem has been solved
        solved.append(ao_est.stats_["solved"])

        #Writes logbook to file
        if save_results:
            log.write("Running adaptive algorithm...\n")
            log.write("\n")
            log.write(str(ao_est.logbook_))
            log.write("\n")
            log.write("============================================")

    #Redesign counts
    avg_redesigns_mut = sum(num_mutation_redesigns) / num_runs
    avg_redesigns_cx = sum(num_crossover_redesigns) / num_runs
    print(f"Number of mutation redesigns: {avg_redesigns_mut}")
    print(f"Number of crossover redesigns: {avg_redesigns_cx}")

    #Mutation similarity score
    if mutation_similarities:
        #Remove runs with empty similarity scores
        cleaned_similarities = [s for s in mutation_similarities if s is not None]

        if cleaned_similarities:
            avg_mutation_similarity = sum(cleaned_similarities) / len(cleaned_similarities)
        else:
            avg_mutation_similarity = None

    if crossover_similarities:
        #Remove runs with empty similarity scores
        cleaned_similarities = [s for s in crossover_similarities if s is not None]

        if cleaned_similarities:
            avg_crossover_similarity = sum(cleaned_similarities) / len(cleaned_similarities)
        else:
            avg_crossover_similarity = None

    print(f"Mutation similarity: {avg_mutation_similarity}")
    print(f"Crossover similarity: {avg_crossover_similarity}")

    #Results on testing data
    avg_testing_fitness = np.mean(all_fit_testing)
    min_testing_fitness = min(all_fit_testing)
    print(f"Average Testing Fitness: {avg_testing_fitness}")
    print(f"Minimum Testing Fitness: {min_testing_fitness}")

    #Find average execution time
    average_exec_time = sum(execution_times) / len(execution_times)
    print(f"Average execution time: {average_exec_time}")

    #Find average number of evaluations
    average_n_evals = sum(n_evals) / len(n_evals)
    print(f"Average number of evaluations: {average_n_evals}")

    #Finds percent of problems solved
    solves_percent = solved.count(True) / len(solved)
    print(f"Percent of problems solved: {solves_percent}")

    #Find success rate of designs
    redesign_success_rate = ao_est.stats_["redesign_success_rate"]
    print(f"Redesign success rate: {redesign_success_rate}")

    #Find overall best operator designs
    if best_mutation_designs:
        best_mutation_design, best_mutation_stats = max(best_mutation_designs, key=lambda x: x[1]["score"])
        best_crossover_design, best_crossover_stats = max(best_crossover_designs, key=lambda x: x[1]["score"])

        #Writes operator designs to files
        mutation_design_file = open(f"{directory_name}/mutation_design.txt", "w")
        crossover_design_file = open(f"{directory_name}/crossover_design.txt", "w")

        mutation_design_file.write(f"Success Rate: {best_mutation_stats["success_rate"]}\n")
        mutation_design_file.write(f"Mean Minimum Fitness Improvement: {best_mutation_stats["min_fitness_improv"]}\n")
        mutation_design_file.write(f"Mean Average Fitness Improvement: {best_mutation_stats["avg_fitness_improv"]}\n")
        mutation_design_file.write(f"\nOperator Design:\n{best_mutation_design}")

        crossover_design_file.write(f"Success Rate: {best_crossover_stats["success_rate"]}\n")
        crossover_design_file.write(f"Mean Minimum Fitness Improvement: {best_crossover_stats["min_fitness_improv"]}\n")
        crossover_design_file.write(f"Mean Average Fitness Improvement: {best_crossover_stats["avg_fitness_improv"]}\n")
        crossover_design_file.write(f"\nOperator Design:\n{best_crossover_design}")

    if save_results:
        #Write to logbook
        log.write("\n")
        log.write("\n")
        log.write(f"Number of mutation redesigns: {avg_redesigns_mut}\n")
        log.write(f"Number of crossover redesigns: {avg_redesigns_cx}\n")

        log.write("\n")
        log.write(f"Mutation similarity: {avg_mutation_similarity}\n")
        log.write(f"Crossover similarity: {avg_crossover_similarity}\n")

        log.write("\n")
        log.write(f"Average Testing Fitness (Standard Operator): {avg_testing_fitness}\n")
        log.write(f"Minimum Testing Fitness (Standard Operator): {min_testing_fitness}\n")

        log.write("\n")
        log.write(f"Average execution time: {average_exec_time}")

        log.write("\n")
        log.write(f"Average number of evaluations: {average_n_evals}")

        log.write("\n")
        log.write(f"Percent of problems solved: {solves_percent}")

        log.write("\n")
        log.write(f"Redesign success rate: {redesign_success_rate}")

        log.close()

    #Update Final Stats
    final_stats["mutation_redesigns"] = avg_redesigns_mut
    final_stats["crossover_redesigns"] = avg_redesigns_cx
    final_stats["mutation_similarity"] = avg_mutation_similarity
    final_stats["crossover_similarity"] = avg_crossover_similarity
    final_stats["min_testing_fitness"] = min_testing_fitness
    final_stats["avg_testing_fitness"] = avg_testing_fitness
    final_stats["avg_execution_time"] = average_exec_time
    final_stats["all_execution_times"] = execution_times
    final_stats["avg_n_evals"] = average_n_evals
    final_stats["solved_percent"] = solves_percent
    final_stats["redesign_success_rate"] = redesign_success_rate
    final_stats["min_training_fitness"] = all_fit_min
    final_stats["min_training_fitness"] = all_fit_min
    final_stats["avg_training_size"] = all_size_avg
    final_stats["all_testing_fitness"] = all_fit_testing
    final_stats["redesign_gens"] = redesign_gens

    return final_stats


def test_configuration(params, tuning_dataset):
    """Tests the effectiveness of a set of parameters on a tuning dataset

    Args:
        params (dict): The model configuration (parameters)
        tuning_dataset ([str]): List of problems to test on 

    Returns:
        [float], [float], [float], [float], [float]: The medians, lower quartiles, upper quartiles, execution times, mutation similarities, crossover similarities for all problems
    """
    medians = []
    q1s = []
    q3s = []
    exec_times = []
    mutation_similarities = []
    crossover_similarities = []

    for problem in tuning_dataset:
        #Perform 5 runs per configuration when tuning
        stats = run_problem_instance(problem, params, None, 5, save_results=False)

        testing_fitnesses = stats["all_testing_fitness"]
        execution_times = stats["all_execution_times"]

        #Return the median fitness from testing
        median, q1, q3 = analyse_across_runs(testing_fitnesses)
        medians.append(median) 
        q1s.append(q1)
        q3s.append(q3)
        exec_times.append(execution_times)
        mutation_similarities.append(stats["mutation_similarity"])
        crossover_similarities.append(stats["crossover_similarity"])

    return medians, q1s, q3s, exec_times, mutation_similarities, crossover_similarities

def hyperparameter_tuning(ranges, tuning_problems, filepath, plot_param=None):
    """Traverses hyperparameter ranges, testing each configuration

    Args:
        ranges (dict): A dictionary of model parameters to be tuned, and the values they can take
        tuning_problems ([str]): The problems used for tuning
        filepath (str): The location to save results
        plot_param (str, optional): The string of a parameter to plot graphs for. Useful when tuning a singular parameter. Defaults to None.
    """
    #Creates list of parameter combinations
    keys = ranges.keys()
    values = ranges.values()

    parameters = [dict(zip(keys, v)) for v in product(*values)]
    parameters_tuples = []
    for param in parameters:
        parameters_tuples.append(tuple(param.items()))

    #Iterates through each parameter combination
    tuning_medians = {}
    tuning_q1s = {}
    tuning_q3s = {}
    tuning_exec_times = {}
    tuning_mut_similarities = {}
    tuning_cross_similarities = {}
    for i in range(len(parameters)):

        #Set up parameters
        current_params = {
            "pop_size": parameters[i]["pop_size"],
            "gens": 10, #TODO: Change
            "max_time": 8.0 * 60.0 * 60.0,
            "cxpb": parameters[i]["cxpb"],
            "mutpb": parameters[i]["mutpb"],
            "tourn_size": parameters[i]["tourn_size"],
            "k": parameters[i]["k"],
            "functions": ["+", "-", "*", "/", "sqrt", "sin", "cos", "log"],
            "verbose": True,
            "self_adapt_req": parameters[i]["self_adapt_req"],
            "default_temperature": parameters[i]["default_temperature"],
            "temperature_alpha": parameters[i]["temperature_alpha"], 
            "maximum_stagnation": 10,
            "model": parameters[i]["model"],
            "reasoning_model": parameters[i]["reasoning_model"]
        }

        param_tuple = parameters_tuples[i]

        medians, q1s, q3s, exec_times, mutation_similarities, crossover_similarities = test_configuration(current_params, tuning_problems)
        tuning_medians[param_tuple] = medians
        tuning_q1s[param_tuple] = q1s
        tuning_q3s[param_tuple] = q3s
        tuning_exec_times[param_tuple] = exec_times
        tuning_mut_similarities[param_tuple] = mutation_similarities
        tuning_cross_similarities[param_tuple] = crossover_similarities

    #Determine average rank for each hyperparameter combination
    medians_list = list(tuning_medians.values())

    parameter_rankings = {param: [] for param in parameters_tuples}
    problem_rankings = []

    #Rank the performance on each problem instance
    for i in range(len(medians_list[0])):
        problem_medians = [medians[i] for medians in medians_list]
        problem_rankings.append(stats.rankdata(problem_medians, method='average'))

    #Gets rank for each problem for each parameter
    for problem in problem_rankings:
        for i in range(len(problem)):
            param = parameters_tuples[i]
            parameter_rankings[param].append(problem[i])

    #Add average rank
    for param, ranks in parameter_rankings.items():
        avg_rank = sum(ranks) / len(ranks)
        parameter_rankings[param].append(avg_rank)

    #Find best configuration
    sorted_ranks = sorted(parameter_rankings.items(), key=lambda x: x[1][-1])

    print("Best configuration")
    print(sorted_ranks[0][0])

    with open(f"{filepath}/tuning_results.txt", "w") as f:
        for param in parameter_rankings:
            f.write(str(param) + "\n")
            f.write(f"Problem 1 Median: {tuning_medians[param][0]}\n")
            f.write(f"Problem 1 Rank: {parameter_rankings[param][0]}\n")
            f.write(f"Problem 2 Median: {tuning_medians[param][1]}\n")
            f.write(f"Problem 2 Rank: {parameter_rankings[param][1]}\n")
            f.write(f"Problem 3 Median: {tuning_medians[param][2]}\n")
            f.write(f"Problem 3 Rank: {parameter_rankings[param][2]}\n")
            f.write(f"Average Rank: {parameter_rankings[param][-1]}\n\n")

            valid = [v for v in tuning_mut_similarities[param] if v is not None]
            mut_mean = np.mean(valid) if valid else 0
            valid = [v for v in tuning_cross_similarities[param] if v is not None]
            cross_mean = np.mean(valid) if valid else 0
            f.write(f"Average Mutation Similarity: {mut_mean}\n")
            f.write(f"Average Crossover Similarity: {cross_mean}\n\n")

        f.write("Best configuration:\n")
        f.write(str(sorted_ranks[0][0]))

    if plot_param:
        #Plot graph for each problem
        for p in range(len(tuning_problems)):
            x = []
            median = []
            q1 = []
            q3 = []

            exec_median = []
            exec_q1 = []
            exec_q3 = []

            #Finds values for lower quartile, upper quartile, and median
            for i in range(len(tuning_medians)):
                parameter = parameters[i]
                parameter_tuple = parameters_tuples[i]
                x.append(parameter[plot_param])
                median.append(tuning_medians[parameter_tuple][p])
                q1.append(tuning_q1s[parameter_tuple][p])
                q3.append(tuning_q3s[parameter_tuple][p])

                exec_median.append(np.median(tuning_exec_times[parameter_tuple][p])) 
                exec_q1.append(np.percentile(tuning_exec_times[parameter_tuple][p], 25))
                exec_q3.append(np.percentile(tuning_exec_times[parameter_tuple][p], 75))

            #Plots fitness graph
            fig = plt.figure(figsize=[7,5])
            ax = plt.subplot(111)
            ax.plot(x, median)
            ax.fill_between(x, q1, q3, alpha=0.2)
            ax.set_xlabel(f"{plot_param} Value")
            ax.set_ylabel("Minimum Testing Fitness")   
            ax.set_title(f"Fitness for {plot_param} Value") 
   
            plt.savefig(f"{filepath}/{plot_param}_fitness_plot_p{p}.pdf", dpi=300)
            plt.show()

            #Plots execution timegraph
            fig = plt.figure(figsize=[7,5])
            ax = plt.subplot(111)
            ax.plot(x, exec_median)
            ax.fill_between(x, exec_q1, exec_q3, alpha=0.2)
            ax.set_xlabel(f"{plot_param} Value")
            ax.set_ylabel("Mean Execution Time (Seconds)")
            ax.set_title(f"Execution Time for {plot_param} Value") 

            plt.savefig(f"{filepath}/{plot_param}_time_plot_p{p}.pdf", dpi=300)
            plt.show()

####MAIN FUNCTIONS####

def tune_gp_model(tuning_ranges, directory_name, plot_param=None):
    """Tunes a GP model with given ranges on 3 problem instances.

    Args:
        tuning_ranges (dict): Parameter ranges
        plot_param (str): The name of the parameter to plot graphs for. Defaults to None.
        directory_name (str): The location to save results
    """
    #Chosen as they cover different areas of the problem space
    problems = ["192_vineyard", "620_fri_c1_1000_25", "201_pol"]

    #Make directory for results
    try:
        os.mkdir(directory_name)
    except FileExistsError:
        pass

    hyperparameter_tuning(tuning_ranges, problems, directory_name, plot_param=plot_param)

def model_comparisons(params, names):
    """Compares multiple models

    Args:
        params ([dict]): The parameters for each model
        names ([str]): The names fo each model
    """

    #Make directory for overall results
    overall_directory_name = f"results/overall_results"
    try:
        os.mkdir(overall_directory_name)
    except FileExistsError:
        pass

    #Choose 8 representative problems
    ground_truth_problems = ["feynman_I_9_18", "feynman_I_6_2a", "feynman_test_10", "feynman_test_10"]
    black_box_problems = ["201_pol", "620_fri_c1_1000_25", "1089_USCrime", "4544_GeographicalOriginalofMusic"]
    all_problems = ground_truth_problems + black_box_problems

    test_problems = ["201_pol", "620_fri_c1_1000_25"]

    #Universal statistics (i.e. across all problems) per model
    mutation_similarities =  {name: [] for name in names[:-1]}
    crossover_similarities = {name: [] for name in names[:-1]}
    redesign_success_rates = {name: [] for name in names[:-1]}

    #Iterates through problems TODO: Update to iterate through all problems
    for problem in test_problems:

        #Make directory for results
        directory_name = f"results/{problem}"
        try:
            os.mkdir(directory_name)
        except FileExistsError:
            pass

        #Statistics 
        training_fitnesses = []
        testing_fitnesses = []
        avg_sizes = []
        solve_rates = []
        avg_n_evals = []
        execution_times = []
        average_execution_times = []

        #Runs the problem with each model
        for i in range(len(params)):
            current_parameters = params[i]
            model_name = names[i]

            stats = run_problem_instance(problem, current_parameters, model_name, num_runs=2, save_results=True) #TODO: Change num runs
            training_fitnesses.append(stats["min_training_fitness"])
            testing_fitnesses.append(stats["all_testing_fitness"])
            avg_sizes.append(stats["avg_training_size"])
            solve_rates.append(stats["solved_percent"] * 100)
            avg_n_evals.append(stats["avg_n_evals"])
            execution_times.append(stats["all_execution_times"])
            average_execution_times.append(stats["avg_execution_time"])

            #Do not calculate similarity for non-LLM based GP
            if i != (len(names) - 1):
                mut_sim = stats["mutation_similarity"]
                cross_sim = stats["crossover_similarity"]
                if mut_sim != None:
                    mutation_similarities[model_name].append(mut_sim)

                if cross_sim != None:
                    crossover_similarities[model_name].append(cross_sim)

                if stats["redesign_success_rate"] is not None:
                    redesign_success_rates[model_name].append((stats["redesign_success_rate"] * 100))

        #Plots
        plot_minimum_fitnesses(names, training_fitnesses, f"{directory_name}/minimum_fitness.pdf")
        plot_boxplot(names, testing_fitnesses, "Testing Fitness per Model", "Minimum Fitness", f"{directory_name}/minimum_fitness_boxplot.pdf")
        plot_avg_size(names, avg_sizes, f"{directory_name}/avg_size.pdf")
        bar_chart("Percent of Problem Instances Solved", "Problem Instances Solved (%)", names, solve_rates, f"{directory_name}/solve_rate.pdf")
        bar_chart("Average Runtime", "Runtime (Seconds)", names, average_execution_times, filepath=f"{directory_name}/runtime_bar.pdf")
        scatter_plot("Minimum Fitness vs Runtime", "Runtime (Seconds)", "Minimum Fitness", names, execution_times, testing_fitnesses)

    #Finds averages across all problems
    mutation_similarities_per_model = [sum(similarities) / len(similarities) for similarities in mutation_similarities.values() if len(similarities) > 0]
    crossover_similarities_per_model = [sum(similarities) / len(similarities) for similarities in crossover_similarities.values() if len(similarities) > 0]
    success_rate_per_model = [sum(sr) / len(sr) for sr in redesign_success_rates.values() if len(sr) > 0]

    if mutation_similarities_per_model and crossover_similarities_per_model:
        similarity_bar_chart(names[:-1], mutation_similarities_per_model, crossover_similarities_per_model, f"{overall_directory_name}/similarity_chart.pdf")

    if success_rate_per_model:
        bar_chart("LLM-Design Success Rate", "Success Rate (%)", names[:-1], success_rate_per_model, f"{overall_directory_name}/success_rate.pdf")
        
def compare_two_approaches(dataset, alg1_name, alg2_name, alg1_params, alg2_params, num_runs=10):
    """Compare 2 approaches, performing statistical testing between them

    Args:
        dataset ([str]): Names of problems to investigate from PMLB dataset
        alg1_name (str): Name of first model
        alg2_name (str): Name of second model
        alg1_params (dict): Parameters of first model
        alg2_params (dict): Parameters of second model
        num_runs (int, optional): Number of iterations of each problem to run. Defaults to 10.
    """
    for problem in dataset:
        #Make directory for results
        directory_name = f"results/{problem}"
        try:
            os.mkdir(directory_name)
        except FileExistsError:
            pass

        traditional_stats = run_problem_instance(problem, alg1_params, alg1_name, num_runs=num_runs)
        adaptive_stats = run_problem_instance(problem, alg2_params, alg2_name, num_runs=num_runs)

        p_value, test_statistic, diff = statistical_testing(traditional_stats["min_testing_fitness"], adaptive_stats["min_testing_fitness"], 0.05)

        log = open(f"{directory_name}/{alg1_name}__{alg2_name}.txt", "w")
        log.write(f"Comparison of {alg1_name} Model against {alg2_name} Model\n\n")
        log.write(f"Wilcoxon Signed Rank Test Statistic:: {test_statistic}\n")
        log.write(f"p-value: {p_value}\n")
        if diff:
            print("Reject the null hypothesis: There is a significant difference between the two samples.")
            log.write("Reject the null hypothesis: There is a significant difference between the two samples.\n")
        else:
            print("Fail to reject the null hypothesis: No significant difference between the two samples.")
            log.write("Fail to reject the null hypothesis: No significant difference between the two samples.\n")
        
def blackbox_vs_groundtruth(optimal_parameters, standard_model_params, model_name):
    """Compares model performance on blackbox versus ground truth problems

    Args:
        optimal_parameters (dict): The parameters of the main model to use
        standard_model_params (dict): The parameters of the model to use without an LLM component
        model_name (str): The name of the optimal model to use
    """
    ground_truth_problems = ["feynman_I_9_18", "feynman_I_6_2a", "feynman_test_10", "feynman_test_5", "feynman_I_18_4", "feynman_II_6_15b", "feynman_III_17_37", "strogatz_barmag2", "strogatz_lv1", "strogatz_predprey1"]
    black_box_problems = ["201_pol", "620_fri_c1_1000_25", "1089_USCrime", "4544_GeographicalOriginalofMusic", "529_pollen", "537_houses", "542_pollution", "1028_SWD", "1029_LEV", "1030_ERA"]

    ground_truth_wins = 0
    ground_truth_losses = 0
    ground_truth_ties = 0

    black_box_wins = 0
    black_box_losses = 0
    black_box_ties = 0

    ground_truth_improvements = []
    black_box_improvements = []

    redesign_gens = []

    #TODO: Do something with improvement
    for problem in ground_truth_problems:
        #Make directory for results
        directory_name = f"results/{problem}"
        try:
            os.mkdir(directory_name)
        except FileExistsError:
            pass

        #TODO: Change num_runs to 10
        adaptive_stats = run_problem_instance(problem, optimal_parameters, model_name, 2, True)
        standard_stats = run_problem_instance(problem, standard_model_params, "No_LLM", 2, True)
        
        #Evaluate based on testing fitnesses
        adaptive_testing_fitnesses = adaptive_stats["all_testing_fitness"]
        standard_testing_fitnesses = standard_stats["all_testing_fitness"]

        adaptive_median = np.median(adaptive_testing_fitnesses) 
        standard_median = np.median(standard_testing_fitnesses)

        improvement_percent = ((standard_median - adaptive_median) / standard_median) * 100
        ground_truth_improvements.append(improvement_percent)

        if adaptive_median < standard_median:
            ground_truth_wins += 1
        elif adaptive_median == standard_median:
            ground_truth_ties += 1
        else:
            ground_truth_losses += 1

    for problem in black_box_problems:
        #Make directory for results
        directory_name = f"results/{problem}"
        try:
            os.mkdir(directory_name)
        except FileExistsError:
            pass

        adaptive_stats = run_problem_instance(problem, optimal_parameters, model_name, 2, True)
        standard_stats = run_problem_instance(problem, standard_model_params, "No_LLM", 2, True)
        
        #Evaluate based on testing fitnesses
        adaptive_testing_fitnesses = adaptive_stats["all_testing_fitness"]
        standard_testing_fitnesses = standard_stats["all_testing_fitness"]

        adaptive_median = np.median(adaptive_testing_fitnesses) 
        standard_median = np.median(standard_testing_fitnesses)

        improvement_percent = ((standard_median - adaptive_median) / standard_median) * 100
        black_box_improvements.append(improvement_percent)
        
        if adaptive_median < standard_median:
            black_box_wins += 1
        elif adaptive_median == standard_median:
            black_box_ties += 1
        else:
            black_box_losses += 1

    compare_problem_types(ground_truth_wins, ground_truth_losses, ground_truth_ties, black_box_wins, black_box_losses, black_box_ties, ground_truth_improvements, black_box_improvements)

#TODO: Clean up this function
if __name__ == "__main__":
    #Set up creator object
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #We want to minimise fitness
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) #Individuals are GP trees (with an associated fitness value)

    #Leave model and reasoning_model as None
    standard_params = {
        "pop_size": 10, #250
        "gens": 15,
        "max_time": 8.0 * 60.0 * 60.0,
        "cxpb": 0.8,
        "mutpb": 0.1,
        "k": 3, 
        "functions": ["+", "-", "*", "/", "sqrt", "sin", "cos", "log"],
        "verbose": True,
        "self_adapt_req": None, #Can be set to None (5 works well)
        "default_temperature": 0.3,
        "temperature_alpha": 0.1,
        "maximum_stagnation": 10,
        "model": None,
        "reasoning_model": False
    }

    testing_params = {
        "pop_size": 10, #250
        "gens": 5,
        "max_time": 8.0 * 60.0 * 60.0,
        "cxpb": 0.8,
        "mutpb": 0.1,
        "k": 5, 
        "functions": ["+", "-", "*", "/", "sqrt", "sin", "cos", "log"],
        "verbose": True,
        "self_adapt_req": None, #Can be set to None (5 works well)
        "default_temperature": 0.3,
        "temperature_alpha": 0.1,
        "maximum_stagnation": 10,
        "model": None,
        "reasoning_model": False
    }

    tuning_ranges = {
        "pop_size": [10], #250
        "cxpb": [0.8],
        "mutpb": [0.1],
        "tourn_size": [3],
        "k": [1,2],
        "self_adapt_req": [None], #Can be set to None (5 works well)
        "default_temperature": [0.3],
        "temperature_alpha": [0.1],
        "model": [None],
        "reasoning_model": [False]
    }

    testing_params_custom_op = {
        "pop_size": 10, #250
        "gens": 15,
        "max_time": 8.0 * 60.0 * 60.0,
        "cxpb": 0.8,
        "mutpb": 0.1,
        "k": 1000, 
        "functions": ["+", "-", "*", "/", "sqrt", "sin", "cos", "log"],
        "verbose": True,
        "self_adapt_req": None, #Can be set to None (5 works well)
        "default_temperature": 0.3,
        "temperature_alpha": 0.1,
        "maximum_stagnation": 10,
        "model": None,
        "reasoning_model": False,
        "custom_crossover_filepath": "docs/default_crossover_design.txt",
        "custom_mutation_filepath": "docs/default_mutation_design.txt"
    }

    models = ["Qwen/Qwen3-Coder-Next-FP8", "zai-org/GLM-5", "moonshotai/Kimi-K2.5", "openai/gpt-oss-120b", "deepseek-ai/DeepSeek-V3.1", "meta-llama/Llama-3.3-70B-Instruct-Turbo", None]
    model_names = ["Qwen3-Coder", "GLM-5", "Kimi K2.5", "GPT OSS 120b", "DeepSeek V3.1", "Llama 3.3 70b", "No LLM"]
    reasoning = [False, True, True, True, True, False, False]

    #Testing
    models = [None, None]
    model_names = ["Standard 1", "Standard 2", "Standard 3", "Standard 4"]
    reasoning = [False, False, False, False]

    # #LLM Testing
    # models = ["openai/gpt-oss-120b", "meta-llama/Llama-3.3-70B-Instruct-Turbo", None]
    # model_names = ["GPT", "Llama", "No LLM"]
    # reasoning = [True, False, False]

    # for i in range(len(models)):
    #     current_parameters = optimal_parameters.copy()
    #     current_parameters["model"] = models[i]
    #     current_parameters["reasoning_model"] = reasoning[i]
    #     model_name = model_names[i]

    # compare_two_approaches(datasets, "GPT-OSS-120b", "Standard", adaptive_params, standard_params, num_runs=num_runs)
    # compare_llms_on_problems(datasets, ["Standard1", "Standard2", "Standard3"], [standard_params, standard_params, standard_params], 3)


    model_comparisons([testing_params_custom_op, testing_params], ["Best-Performing Operator", "Standard GP"])

# Results:
# 1. Tune baseline GP (no LLM) - tune_gp_model
# 2. Tune k (plot k) - tune_gp_model
# 3. Tune self-adapt + default temp (no plot) - tune_gp_model
# 4. Compare multiple models - model_comparisons(params, names)
# 5. Black box vs Ground Truth for optimal setup - blackbox_vs_groundtruth(optimal_parameters, standard_model_params, model_name)
# 6. Compare optimal LLM-GP against Standard GP - compare_two_approaches(dataset, alg1_name, alg2_name, alg1_params, alg2_params, num_runs=10)
# 7. Compare optimal LLM-GP against GP with Fixed Operator Design - compare_two_approaches(dataset, alg1_name, alg2_name, alg1_params, alg2_params, num_runs=10)
# 8. Compare Fixed Operator Design GP against Standard GP - compare_two_approaches(dataset, alg1_name, alg2_name, alg1_params, alg2_params, num_runs=10)

