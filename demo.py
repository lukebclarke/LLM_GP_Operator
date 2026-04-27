from deap import algorithms, base, creator, tools, gp
from adaptive_operators.gp_model import AdaptiveRegressor
from pmlb import fetch_data, dataset_names
import pygraphviz as pgv
import gradio as gr
import os

def plot_individual(individual, filename):
    nodes, edges, labels = gp.graph(individual)

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw(filename)

def setup_demo():
    #Set up creator object
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #We want to minimise fitness
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) #Individuals are GP trees (with an associated fitness value)

    optimal_parameters = {
        "pop_size": 3, 
        "gens": 1,
        "max_time": 8.0 * 60.0 * 60.0,
        "cxpb": 0.9,
        "mutpb": 0.05,
        "tourn_size": 7,
        "k": 5, 
        "functions": ["+", "-", "*", "/", "sqrt", "sin", "cos", "ln", "^2"],
        "verbose": True,
        "self_adapt_req": 5,
        "default_temperature": 0.3,
        "temperature_alpha": 0.1,
        "maximum_stagnation": 100,
        "model": "openai/gpt-oss-120b",
        "reasoning_model": True
    }

    #Example problem to get X and y - not actually used in Demo 
    X, Y = fetch_data("nikuradse_2", return_X_y=True)

    #Set up regressor
    ao_est = AdaptiveRegressor(**optimal_parameters)
    pset = ao_est.create_pset(1)
    ao_est.setup_daytona(max_attempts=10)
    ao_est.create_toolbox(X, Y)

    #Updates prompts for demo
    with open("docs/LLMPromptCrossoverDemo.txt", "r") as f:
        ao_est.custom_crossover.llm_prompt = f.read()

    with open("docs/LLMPromptMutationDemo.txt", "r") as f:
        ao_est.custom_mutate.llm_prompt = f.read()

    ind1 = gp.PrimitiveTree.from_string("sub(1, add(mul(ARG0, -1), mul(1, ARG0)))", pset)
    ind2 = gp.PrimitiveTree.from_string("add(mul(pi, ARG0), square(ARG0))", pset)

    plot_individual(ind1, "temp/parent1.png")
    plot_individual(ind2, "temp/parent2.png")

    return ao_est

def get_operator_designs(ao_est):
    #TODO: Get saved designs if it fails after 5 seconds
    ao_est.custom_mutate.redesign_operator()
    ao_est.custom_crossover.redesign_operator()

    mutation_design = ao_est.custom_mutate.operator_design
    crossover_design = ao_est.custom_crossover.operator_design

    print(mutation_design)
    print(crossover_design)

    return mutation_design, crossover_design

def demo():
    with gr.Blocks() as demo:
        parent1_path = "temp/parent1.png"
        parent2_path = "temp/parent2.png"

        with gr.Walkthrough(selected=1) as walkthrough:
            #Step 1 - Show parents
            with gr.Step("Step 1", id=1):

                with gr.Row():
                    with gr.Column(scale=1):
                        p1 = gr.Image(parent1_path, height=400)
                    with gr.Column(scale=1):
                        p2 = gr.Image(parent2_path, height=400)
                
                btn = gr.Button("Generate Genetic Operators")
                btn.click(lambda: gr.Walkthrough(selected=2), outputs=walkthrough)

            #Step 2 - Generate genetic operators
            with gr.Step("Step 2", id=2):
                pass
            #Step 3 - Applying genetic operators remotely
            with gr.Step("Step 3", id=2):
                txt = gr.Textbox("Generated Offspring")

        demo.launch()

ao_est = setup_demo()
get_operator_designs(ao_est)
# demo()

