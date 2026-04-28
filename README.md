# LLM_GP_Operator
Creates LLM-Guided Genetic Operators to enhance the crossover and mutation operations within Genetic Programming.

How to run:
- Create a virtual environment and run the requirements.txt
- If the PMLB dataset is not installed from the requirements.txt, instead install it from https://github.com/EpistasisLab/pmlb, and run `python3 -m pip install path/to/pmlb`. Using `python3 pip install pmlb` is not sufficient for this project since it uses datasets not included in the Python library
- Set up the `TOGETHER_AI` and `DAYTONA_API_KEY` environment variables with your Together AI API and Daytona API keys, respectively
- To run all experiments, use the experiments_walkthrough.ipynb notebook file
- To deploy as a Docker container to run on a remote virtual machine, place an experiment in the experiments.py file, and build the container using the provided Dockerfile.