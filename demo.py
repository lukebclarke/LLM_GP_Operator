from deap import algorithms, base, creator, tools, gp
from adaptive_operators.gp_model import AdaptiveRegressor
from pmlb import fetch_data, dataset_names
import pygraphviz as pgv
import gradio as gr
import os
import random

#Demo can be run locally using previously saved LLM-generated operators
LOCAL_EXECUTION = True

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

    if not LOCAL_EXECUTION:
        ao_est.setup_daytona(max_attempts=10)

    pset = ao_est.create_pset(1)
    ao_est.pset = pset
    ao_est.create_toolbox(X, Y)

    #Updates prompts for demo
    with open("docs/LLMPromptCrossoverDemo.txt", "r") as f:
        ao_est.custom_crossover.llm_prompt = f.read()

    with open("docs/LLMPromptMutationDemo.txt", "r") as f:
        ao_est.custom_mutate.llm_prompt = f.read()

    ind1 = gp.PrimitiveTree.from_string("add(mul(ARG0, protectedDiv(add(ARG0, 1), square(ARG0))), sin(sub(ARG0, pi)))", pset)
    ind2 = gp.PrimitiveTree.from_string("sub(add(mul(ARG0, ARG0), pi), protectedLog(add(ARG0, 1)))", pset)

    plot_individual(ind1, "temp/parent1.png")
    plot_individual(ind2, "temp/parent2.png")

    return ao_est, ind1, ind2

def get_mutation_design(ao_est):
    #TODO: Get saved designs if it fails after 5 seconds
    if LOCAL_EXECUTION:
        files = os.listdir("demo_designs/mutation_designs")
        mutation_design_filepath = random.choice(files)
        with open("demo_designs/mutation_designs/" + mutation_design_filepath, "r") as f:
            mutation_design = f.read()
            ao_est.custom_mutate.operator_design = mutation_design
    else:
        ao_est.custom_mutate.redesign_operator()
        mutation_design = ao_est.custom_mutate.operator_design

    return mutation_design

def get_crossover_design(ao_est):
    if LOCAL_EXECUTION:
        files = os.listdir("demo_designs/crossover_designs")
        crossover_design_filepath = random.choice(files)
        with open("demo_designs/crossover_designs/" + crossover_design_filepath, "r") as f:
            crossover_design = f.read()
            ao_est.custom_crossover.operator_design = crossover_design
    else:
        ao_est.custom_crossover.redesign_operator()
        crossover_design = ao_est.custom_crossover.operator_design

    return crossover_design

def generate_operators(ao_est):
    mutation_design = get_mutation_design(ao_est)
    crossover_design = get_crossover_design(ao_est)

    return mutation_design, crossover_design

def apply_mutation(ao_est, individual):
    #Allow 10 attempts
    for i in range(10):
        try:
            if LOCAL_EXECUTION:
                offspring = ao_est.custom_mutate.llm_custom_operator_locally([individual])[0]
            else:
                offspring = ao_est.custom_mutate.llm_custom_operator_daytona([individual])[0]
        except:
            if i < 5:
                continue

            raise Exception("Too many retries")

    print(f"OFFSPRING: {offspring}")
    filepath = f"temp/mutation_offspring.png"
    plot_individual(offspring, filepath)

    return filepath

def apply_crossover(ao_est, individual1, individual2):
    #Allow 10 attempts
    for i in range(10):
        try:
            if LOCAL_EXECUTION:
                offspring = ao_est.custom_crossover.llm_custom_operator_locally([individual1, individual2])
            else:
                offspring = ao_est.custom_crossover.llm_custom_operator_daytona([individual1, individual2])
        except:
            if i < 5:
                continue

            raise Exception("Too many retries")

    plot_individual(offspring[0], "temp/cx_offspring1.png")
    plot_individual(offspring[1], "temp/cx_offspring2.png")

    return "temp/cx_offspring1.png", "temp/cx_offspring2.png"

def demo():
    ao_est, ind1, ind2 = setup_demo()
    parent1_path = "temp/parent1.png"
    parent2_path = "temp/parent2.png"

    with gr.Blocks() as demo:

        with gr.Walkthrough(selected=1) as walkthrough:
            #Step 1 - Show parents
            with gr.Step("Current Population", id=1):

                with gr.Row():
                    with gr.Column(scale=1):
                        p1 = gr.Image(parent1_path, height=400)
                    with gr.Column(scale=1):
                        p2 = gr.Image(parent2_path, height=400)
                
                btn1 = gr.Button("Generate Genetic Operators")

            #Step 2 - Generate genetic operators
            with gr.Step("Generate Operators", id=2):
                with gr.Row():
                    with gr.Column(scale=1):
                        mut_button = gr.Button("Apply Mutation Operator")
                        mut_retry = gr.Button("Regenerate")

                        mutation_code = gr.Code(value=get_mutation_design(ao_est), label="Custom Mutation Design")
                    with gr.Column(scale=1):
                        cx_button = gr.Button("Apply Crossover Operator")
                        cx_retry = gr.Button("Regenerate")

                        crossover_code = gr.Code(value=get_crossover_design(ao_est), label="Custom Crossover Design")

                #Generates operators and advances to next screen
                gr.on(
                    triggers=[btn1.click],
                    fn=lambda: gr.Walkthrough(selected=2), outputs=walkthrough,
                ).then(
                    fn=lambda: generate_operators(ao_est),
                    inputs=[],
                    outputs=[mutation_code, crossover_code]
                )

                mut_retry.click(lambda: get_mutation_design(ao_est), outputs=mutation_code)
                cx_retry.click(lambda: get_crossover_design(ao_est), outputs=crossover_code)

                #Regenerate operators
                gr.on(
                    triggers=[mut_retry.click],
                    fn=lambda: get_mutation_design(ao_est),
                    inputs=[],
                    outputs=[mutation_code],
                )
            
            #Step 3 - Applying mutation operator remotely
            with gr.Step("Mutation", id=3):
                with gr.Row():
                    with gr.Column(scale=1):
                        mut_parent = gr.Image(parent1_path, height=400, label="Parent", type="filepath")
                    with gr.Column(scale=1):
                        mut_offspring = gr.Image(height=400, label="Offspring", type="filepath")

                return1 = gr.Button("View Designs")

                #View mutation results
                gr.on(
                    triggers=[mut_button.click],
                    fn=lambda: gr.Walkthrough(selected=3), 
                    outputs=walkthrough
                ).then(
                    fn=lambda: apply_mutation(ao_est, ind1),
                    outputs=mut_offspring
                )

                return1.click(fn=lambda: gr.Walkthrough(selected=2), outputs=walkthrough)

            #Step 4 - Applying crossover operator remotely
            with gr.Step("Crossover", id=4):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        cx_parent1 = gr.Image(parent1_path, height=400, label="Parent 1", type="filepath")
                    with gr.Column(scale=1):
                        cx_parent2 = gr.Image(parent2_path, height=400, label="Parent 2", type="filepath")

                with gr.Row():
                    with gr.Column(scale=1):
                        cx_offspring_1 = gr.Image(height=400, label="Offspring 1", type="filepath")
                    with gr.Column(scale=1):
                        cx_offspring_2 = gr.Image(height=400, label="Offspring 2", type="filepath")

                return2 = gr.Button("View Designs")

                #View crossover results
                gr.on(
                    triggers=[cx_button.click],
                    fn=lambda: gr.Walkthrough(selected=4), 
                    outputs=walkthrough
                ).then(
                    fn=lambda: apply_crossover(ao_est, ind1, ind2),
                    outputs=[cx_offspring_1, cx_offspring_2]
                )

                return2.click(fn=lambda: gr.Walkthrough(selected=2), outputs=walkthrough)


        demo.launch()

if __name__ == "__main__":
    demo()
