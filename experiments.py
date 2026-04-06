from adaptive_operators.gp_model import AdaptiveRegressor
from standard_operators.gp_model import StandardRegressor
from adaptive_operators.base_operator import MaximumNumberRetries

from pmlb import fetch_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
import random
import math

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from sklearn.model_selection import train_test_split

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


def run_problem_instance(problem_name, params, num_runs=10):
    X, Y = fetch_data(problem_name, return_X_y=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #Make directory for results
    directory_name = f"results/{problem_name}"
    try:
        os.mkdir(directory_name)
    except FileExistsError:
        pass

    #Opens log file
    log = open(f"{directory_name}/logbook.txt", "w")

    #Create new AdaptiveRegressor for each problem instance
    ao_est = AdaptiveRegressor(**params)

    #Statistics
    all_size_avg_ea = []
    all_fit_min_ea = []
    all_fit_testing_ea = []

    all_size_avg_ao = []
    all_fit_min_ao = []
    all_fit_testing_ao = []

    num_mutation_redesigns = []
    num_crossover_redesigns = []

    best_mutation_designs = []
    best_crossover_designs = []

    for i in range(num_runs):
        #Defines random seed
        params["random_state"] = 42 + i
        
        #TODO: Prevent this from crashing altogether
        for i in range(10):
            #Run adaptive evolutionary algorithm without re-initialising sanbdox
            try:
                ao_est = AdaptiveRegressor(**params)
                ao_est.fit(X_train, y_train)
                break
            except MaximumNumberRetries:
                continue

        ao_est.shutdown_sandbox()

        #Run standard evolutionary algorithm
        print("Running standard EA")
        standard_est = StandardRegressor(pop_size=params["pop_size"], gens=params["gens"], max_time=params["max_time"], cxpb=params["cxpb"], mutpb=params["mutpb"], functions=params["functions"], verbose=params["verbose"], random_state=params["random_state"])
        standard_est.fit(X_train, y_train)

        #Get fitnesses on testing data
        standard_test = standard_est.predict(X_test)
        ao_test = ao_est.predict(X_test)

        #Evaluate the mean squared error between the expression and the real function
        all_fit_testing_ea.append(np.mean((standard_test - y_test) ** 2))
        all_fit_testing_ao.append(np.mean((ao_test - y_test) ** 2))

        #Update statistics
        all_size_avg_ea.append(standard_est.logbook_.chapters["size"].select("avg"))
        all_fit_min_ea.append(standard_est.logbook_.chapters["fitness"].select("min"))

        all_size_avg_ao.append(ao_est.logbook_.chapters["size"].select("avg"))
        all_fit_min_ao.append(ao_est.logbook_.chapters["fitness"].select("min"))

        num_mutation_redesigns.append(ao_est.stats_["num_mutation_redesigns"])
        num_crossover_redesigns.append(ao_est.stats_["num_crossover_redesigns"])

        redesign_gens = ao_est.stats_["redesign_generations"]
        fitness_improvement_per_gen_ao = ao_est.stats_["fitness_improvements"]
        fitness_improvement_per_gen_standard = standard_est.stats_["fitness_improvements"]

        best_mutation_design = ao_est.stats_["best_mutation_design"]
        best_crossover_design = ao_est.stats_["best_crossover_design"]
        if best_mutation_design:
            best_mutation_designs.append()

        if best_crossover_design:
            best_crossover_designs.append()

        graph_name = f"/fitness_improvement_run{i}"
        graph_filepath = directory_name + graph_name
        plot_improvement_graph("Fitness Improvement", "Standard", "Adaptive Operator", fitness_improvement_per_gen_standard, fitness_improvement_per_gen_ao, redesign_gens, filepath=graph_filepath)
        plot_improvement_graph_solo("Fitness Improvement", fitness_improvement_per_gen_ao, redesign_gens, filepath=graph_filepath + "_solo")

        #Write to logbook
        log.write("Running standard algorithm...\n")
        log.write("\n")
        log.write(str(standard_est.logbook_))
        log.write("\n")
        log.write("\n")
        log.write("Running adaptive algorithm...\n")
        log.write("\n")
        log.write(str(ao_est.logbook_))
        log.write("\n")
        log.write("============================================")

    #Gets statistics across all runs
    ea_size_avg, ea_fit_min = get_stats(all_size_avg_ea, all_fit_min_ea)
    ao_size_avg, ao_fit_min = get_stats(all_size_avg_ao, all_fit_min_ao)

    #Redesign counts
    avg_redesigns_mut = sum(num_mutation_redesigns) / num_runs
    avg_redesigns_cx = sum(num_crossover_redesigns) / num_runs
    print(f"Number of mutation redesigns: {avg_redesigns_mut}")
    print(f"Number of crossover redesigns: {avg_redesigns_cx}")
    log.write("\n")
    log.write("\n")
    log.write(f"Number of mutation redesigns: {avg_redesigns_mut}\n")
    log.write(f"Number of crossover redesigns: {avg_redesigns_cx}\n")

    #Results on testing data
    print(f"Average Testing Fitness (Standard Operator): {np.mean(all_fit_testing_ea)}")
    print(f"Average Testing Fitness (Adaptive Operator): {np.mean(all_fit_testing_ao)}")
    print(f"Minimum Testing Fitness (Standard Operator): {min(all_fit_testing_ea)}")
    print(f"Minimum Testing Fitness (Adaptive Operator): {min(all_fit_testing_ao)}")
    log.write("\n")
    log.write("\n")
    log.write(f"Average Testing Fitness (Standard Operator): {np.mean(all_fit_testing_ea)}\n")
    log.write(f"Average Testing Fitness (Adaptive Operator): {np.mean(all_fit_testing_ao)}\n")
    log.write(f"Minimum Testing Fitness (Standard Operator): {min(all_fit_testing_ea)}\n")
    log.write(f"Minimum Testing Fitness (Adaptive Operator): {min(all_fit_testing_ao)}\n")

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

    #Save graphs to results folder
    graph_file = f"{directory_name}/average_size.pdf"
    plot_comparison_graph("Average Size", "Standard", "Adaptive Operator", ea_size_avg, ao_size_avg, filepath=graph_file)
    graph_file = f"{directory_name}/min_fitness.pdf"
    plot_comparison_graph("Minimum Fitness", "Standard", "Adaptive Operator", ea_fit_min, ao_fit_min, filepath=graph_file)

    log.close()

def main():
    #Parameters
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #We want to minimise fitness
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) #Individuals are GP trees (with an associated fitness value)

    problem_list = "problems/ground_truth.txt"
    num_runs = 1
    params = {
        "pop_size": 10, #250
        "gens": 15,
        "max_time": 8.0 * 60.0 * 60.0,
        "cxpb": 0.8,
        "mutpb": 0.1,
        "k": 5,
        "functions": ["+", "-", "*", "/", "sqrt", "sin", "cos", "log"],
        "verbose": True,
        "self_adapt_req": 5,
        "default_temperature": 0.3,
        "temperature_alpha": 0.1,
        "maximum_stagnation": 1
    }

    #Finds all ground truth datasets
    with open(problem_list, "r") as f:
        problems = [line.strip() for line in f if line.strip()]

    #Chooses 10 random problems
    datasets = random.sample(problems, 10)

    for problem in datasets:
        run_problem_instance(problem, params, num_runs=num_runs)
    
if __name__ == "__main__":
    main() 