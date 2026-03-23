from adaptive_regressor import AdaptiveRegressor
from standard_regressor import StandardRegressor

from pmlb import fetch_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 

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

def plot_improvement_graph(metric_name, values, redesign_generations, filepath=None):
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

    all_size_avg_ao = []
    all_fit_min_ao = []

    mutation_redesigns = []
    crossover_redesigns = []

    for i in range(num_runs):
        print("Run number: ", i)

        #Run standard evolutionary algorithm
        print("Running standard EA")
        standard_est = StandardRegressor(**params)
        standard_est.fit(X, Y)
        
        #Run adaptive evolutionary algorithm without re-initialising sanbdox
        ao_est.fit(X, Y)

        #Update statistics
        all_size_avg_ea.append(standard_est.logbook_.chapters["size"].select("avg"))
        all_fit_min_ea.append(standard_est.logbook_.chapters["fitness"].select("min"))

        all_size_avg_ao.append(ao_est.logbook_.chapters["size"].select("avg"))
        all_fit_min_ao.append(ao_est.logbook_.chapters["fitness"].select("min"))

        mutation_redesigns.append(ao_est.stats_["mutation_redesigns"])
        crossover_redesigns.append(ao_est.stats_["crossover_redesigns"])

        redesign_gens = ao_est.stats_["redesign_generations"]
        fitness_improvement_per_gen = ao_est.stats_["fitness_improvements"]

        graph_name = f"/fitness_improvement_run{i}"
        graph_filepath = directory_name + graph_name
        plot_improvement_graph("Fitness Improvement", fitness_improvement_per_gen, redesign_gens, filepath=graph_filepath)

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

    ao_est.shutdown_sandbox()

    #Gets statistics across all runs
    ea_size_avg, ea_fit_min = get_stats(all_size_avg_ea, all_fit_min_ea)
    ao_size_avg, ao_fit_min = get_stats(all_size_avg_ao, all_fit_min_ao)

    #Redesign counts
    avg_redesigns_mut = sum(mutation_redesigns) / num_runs
    avg_redesigns_cx = sum(crossover_redesigns) / num_runs
    print(f"Number of mutation redesigns: {avg_redesigns_mut}")
    print(f"Number of crossover redesigns: {avg_redesigns_cx}")
    log.write(f"Number of mutation redesigns: {avg_redesigns_mut}\n")
    log.write(f"Number of crossover redesigns: {avg_redesigns_cx}\n")

    #Graphs
    graph_file = f"{directory_name}/average_size.pdf"
    plot_comparison_graph("Average Size", "Standard", "Adaptive Operator", ea_size_avg, ao_size_avg, filepath=graph_file)
    graph_file = f"{directory_name}/min_fitness.pdf"
    plot_comparison_graph("Minimum Fitness", "Standard", "Adaptive Operator", ea_fit_min, ao_fit_min, filepath=graph_file)

    log.close()

def main():
    dataset_name = 'feynman_II_38_14'
    num_runs = 1

    params = {
        "pop_size": 20,
        "gens": 2,
        "max_time": 8.0 * 60.0 * 60.0,
        "cxpb": 0.7,
        "mutpb": 0.1,
        "k": 3,
        "functions": ["+", "-", "*", "/", "sqrt", "sin", "cos", "log"],
        "verbose": True
    }

    run_problem_instance(dataset_name, params, num_runs=num_runs)

    
if __name__ == "__main__":
    main() 