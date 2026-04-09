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

import scipy.stats as stats

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from sklearn.model_selection import train_test_split

from matplotlib.colors import TABLEAU_COLORS, same_color

def get_stats(all_size_avg, all_fit_min):
    #Ensure all runs are of same length (e.g. padding)
    max_length_alg = max((len(run)) for run in all_fit_min)

    padded_avg_size = []
    padded_min_fit = []

    #Finds padded metrics for each run
    for i in range(len(all_fit_min)):
        padded_avg_size.append(all_size_avg[i] + ([0] * (max_length_alg - len(all_size_avg[i]))))
        padded_min_fit.append(all_fit_min[i] + ([0] * (max_length_alg - len(all_fit_min[i]))))

    #Convert to numpy to find average
    all_size_avg_runs = np.array(padded_avg_size)
    all_fit_min_runs = np.array(padded_min_fit)

    #Find mean of runs across each generation
    size_avg_mean = np.mean(all_size_avg_runs, axis=0)
    fit_min_mean = np.mean(all_fit_min_runs, axis=0)

    return size_avg_mean, fit_min_mean

def pad_runs_multiple_algorithms(algorithm_runs):
    """Pads runs from multiple algorithms

    Args:
        algorithm_runs ([[[float]]]): An array containing a list of each algorithms, and the runs within that algorithm

    Returns:
        _type_: _description_
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

def analyse_minimum_fitness_across_runs(minimum_fitnesses):
    #Runs must be padded already
    minimum_fitnesses = np.array(minimum_fitnesses)

    medians = np.median(minimum_fitnesses, axis=0)
    q1s = np.percentile(minimum_fitnesses, 25, axis=0)
    q3s = np.percentile(minimum_fitnesses, 75, axis=0)

    return medians, q1s, q3s

def plot_minimum_fitnesses(alg_names, alg_min_fitnesses, filepath=None):
    #Pad the minimum fitnesses (so each run is the same length)
    runs = np.array(pad_runs_multiple_algorithms(alg_min_fitnesses))
    print("RUNS")
    print(runs)

    fig = plt.figure(figsize=[7,5])
    ax = plt.subplot(111)
    for i in range(len(runs)):

        median, q1, q3 = analyse_minimum_fitness_across_runs(runs[i])
        print(median)
        x = np.arange(len(median))
        ax.plot(x, median, label=alg_names[i])
        ax.fill_between(x, q1, q3, alpha=0.2)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Minimum Fitness")
        ax.legend()
    if filepath:
        plt.savefig(filepath, dpi=300)
    plt.show()

def box_plot_min_fitnesses(names, minimum_fitnesses, filepath=None):
    fig = plt.figure(figsize=[7,5])
    ax = plt.subplot(111)

    #Generates box plot
    bplot = ax.boxplot(minimum_fitnesses, tick_labels=names, patch_artist=True)

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

    ax.set_title('Minimum Fitness')

    if filepath:
        plt.savefig(filepath, dpi=300)
    plt.show()


def plot_improvement_graph(metric_name, alg1_label, alg2_label, values_alg1, values_alg2, redesign_generations, filepath=None):
    gens_alg1 = list(range(0, len(values_alg1), 1))
    gens_alg2 = list(range(0, len(values_alg2), 1))

    fig = plt.figure(figsize=[7,5])
    ax = plt.subplot(111)
    ax.plot(gens_alg1, values_alg1, label=alg1_label) 
    ax.plot(gens_alg2, values_alg2, label=alg2_label)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, max(len(gens_alg1), len(gens_alg2)), 5))
    ax.set_xlabel("Generations")
    ax.set_ylabel(metric_name)
    ax.legend()

    #Adds references to redesign generations
    for gen in redesign_generations:
        ax.axvline(x=gen, linestyle='--')

    if filepath:
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_improvement_graph_solo(metric_name, values, redesign_generations, filepath=None):
    gens = list(range(0, len(values), 1))

    fig = plt.figure(figsize=[7,5])
    ax = plt.subplot(111)
    ax.plot(gens, values) 
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, len(gens), 5))
    ax.set_xlabel("Generations")
    ax.set_ylabel(metric_name)

    #Adds references to redesign generations
    for gen in redesign_generations:
        ax.axvline(x=gen, linestyle='--')

    if filepath:
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

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

def plot_comparison_graph(metric_name, alg1_label, alg2_label, metric_values1, metric_values2, filepath=None):
    xdata1 = list(range(0, len(metric_values1), 1))
    ydata1 = metric_values1

    xdata2 = list(range(0, len(metric_values2), 1))
    ydata2 = metric_values2

    fig = plt.figure(figsize=[7,5])
    ax = plt.subplot(111)
    ax.plot(xdata1, ydata1, label=alg1_label) 
    ax.plot(xdata2, ydata2, label=alg2_label) 
    ax.set_xlabel("Generations")
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.grid('on')

    if filepath:
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def run_problem_instance(problem_name, params, model_name, num_runs=10):
    #Gets training and testing data
    X, Y = fetch_data(problem_name, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #Make for model within results
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
    log.write("\n")
    log.write("\n")
    log.write(f"Number of mutation redesigns: {avg_redesigns_mut}\n")
    log.write(f"Number of crossover redesigns: {avg_redesigns_cx}\n")

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
    log.write("\n")
    log.write(f"Mutation similarity: {avg_mutation_similarity}\n")
    log.write(f"Crossover similarity: {avg_crossover_similarity}\n")

    #Results on testing data
    avg_testing_fitness = np.mean(all_fit_testing)
    min_testing_fitness = min(all_fit_testing)
    print(f"Average Testing Fitness: {avg_testing_fitness}")
    print(f"Minimum Testing Fitness: {min_testing_fitness}")
    log.write("\n")
    log.write(f"Average Testing Fitness (Standard Operator): {avg_testing_fitness}\n")
    log.write(f"Minimum Testing Fitness (Standard Operator): {min_testing_fitness}\n")

    #Find average execution time
    average_exec_time = sum(execution_times) / len(execution_times)
    print(f"Average execution time: {average_exec_time}")
    log.write("\n")
    log.write(f"Average execution time: {average_exec_time}")

    #Find average number of evaluations
    average_n_evals = sum(n_evals) / len(n_evals)
    print(f"Average number of evaluations: {average_n_evals}")
    log.write("\n")
    log.write(f"Average number of evaluations: {average_n_evals}")

    #Finds percent of problems solved
    solves_percent = solved.count(True) / len(solved)
    print(f"Percent of problems solved: {solves_percent}")
    log.write("\n")
    log.write(f"Percent of problems solved: {solves_percent}")

    #Find success rate of designs
    redesign_success_rate = ao_est.stats_["redesign_success_rate"]
    print(f"Redesign success rate: {redesign_success_rate}")
    log.write("\n")
    log.write(f"Redesign success rate: {redesign_success_rate}")

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

    #Update Final Stats
    final_stats["mutation_redesigns"] = avg_redesigns_mut
    final_stats["crossover_redesigns"] = avg_redesigns_cx
    final_stats["mutation_similarity"] = avg_mutation_similarity
    final_stats["crossover_similarity"] = avg_crossover_similarity
    final_stats["min_testing_fitness"] = min_testing_fitness
    final_stats["avg_testing_fitness"] = avg_testing_fitness
    final_stats["avg_execution_time"] = average_exec_time
    final_stats["avg_n_evals"] = average_n_evals
    final_stats["solved_percent"] = solves_percent
    final_stats["redesign_success_rate"] = redesign_success_rate
    final_stats["min_training_fitness"] = all_fit_min
    final_stats["min_training_fitness"] = all_fit_min
    final_stats["avg_training_size"] = all_size_avg
    final_stats["all_testing_fitness"] = all_fit_testing

    log.close()

    return final_stats

def compare_two_approaches(dataset, alg1_name, alg2_name, alg1_params, alg2_params, num_runs=10):
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

def compare_llms_on_problems(dataset, names, model_params, num_runs=10):
    for problem in dataset:
        #Make directory for results
        directory_name = f"results/{problem}"
        try:
            os.mkdir(directory_name)
        except FileExistsError:
            pass

        training_fitnesses = []
        testing_fitnesses = []
        for i in range(len(model_params)):
            stats = run_problem_instance(problem, model_params[i], names[i], num_runs=num_runs)
            training_fitnesses.append(stats["min_training_fitness"])
            testing_fitnesses.append(stats["all_testing_fitness"])

        plot_minimum_fitnesses(names, training_fitnesses, f"{directory_name}/minimum_fitness.pdf")
        box_plot_min_fitnesses(names, testing_fitnesses, f"{directory_name}/minimum_fitness_boxplot.pdf")

def main():
    #Parameters
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #We want to minimise fitness
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) #Individuals are GP trees (with an associated fitness value)

    problem_list = "problems/ground_truth.txt"
    num_runs = 1
    num_problems = 1

    adaptive_params = {
        "pop_size": 10, #250
        "gens": 10,
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
        "model": "openai/gpt-oss-120b",
        "reasoning_model": True
    }

    standard_params = {
        "pop_size": 10, #250
        "gens": 10,
        "max_time": 8.0 * 60.0 * 60.0,
        "cxpb": 0.8,
        "mutpb": 0.1,
        "k": 100000,
        "functions": ["+", "-", "*", "/", "sqrt", "sin", "cos", "log"],
        "verbose": True,
        "self_adapt_req": None, #Can be set to None (5 works well)
        "default_temperature": 0.3,
        "temperature_alpha": 0.1,
        "maximum_stagnation": 10,
        "model": None,
        "reasoning_model": False
    }

    #Finds all ground truth datasets
    with open(problem_list, "r") as f:
        problems = [line.strip() for line in f if line.strip()]

    #Chooses 10 random problems
    datasets = random.sample(problems, num_problems)

    # compare_two_approaches(datasets, "GPT-OSS-120b", "Standard", adaptive_params, standard_params, num_runs=num_runs)
    compare_llms_on_problems(datasets, ["Standard1", "Standard2", "Standard3"], [standard_params, standard_params, standard_params], 3)
    
if __name__ == "__main__":
    main() 