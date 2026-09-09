# Replication Package — SIMULATING COMPETITIVE COLLOCATION STRATEGIES WITH DEEP REINFORCEMENT LEARNING
This simulation code replicates the simulation data for the paper “Simulating competitive collocation strategies with deep reinforcement learning” and the respective analyses. To execute the simulation, download all files as one zip, unzip, and follow the instructions in the README file outlining the steps.

## Requirements

- Python 3.10
- Poetry

## Basic Setup (execute once)
Save the zip file into a folder, extract and execute from folder:

    # Install poetry
    pip install poetry

    # Add shell plug-in (optional but recommended)
    poetry self add poetry-plugin-shell

    # Set-up venv (creates a virtual environment and installs all dependencies in the pyproject.toml file)
    poetry install    

## Running the simulation (from folder)

    # Activate venv (line below works only with plug-in)
    poetry shell

    # Execute simulation
    python train.py

    # Execute single seed analysis (optional)
    python single_seeds-jibs.py

    # Close venv
    deactivate 

## Project structure

    ├── train.py                    # Master script
    ├── firm_world.py               # Contains firm-world
    ├── ppo_cnn_agent.py            # Neural Network (CNN)
    ├── clean_ppo_selfplay.py       # DRL-execution
    ├── selfplay_utils.py           # Helper/Utility functions
    ├── fixed_profit_max.py         # Computes comparison baseline
    ├── merge_csv.py                # Merges all seeds into one file
    ├── single_seeds-jibs.py        # Creates single-seed analysis plots
    ├── collocation_figures-jibs.R  # Creates collocation and return figures from the paper (can be executed separately after simulation)
    ├── pyproject.toml              # Dependency specification
    ├── poetry.lock                 # Exact dependency versions (do not edit)
    └── simulation/                 # Output folder (created  by train.py)
        ├── 0,....,[max(Seeds)]         # Result for each seed is stored in a separate folder
            ├── models/                     # Model data
            └── results/                    # Results for a single seed
        ├── learning_curves/            # Contains learnings curves for each seed
        ├── results/                    # Final merged results
            └── shap/                       # Folder (created by single_seed-jibs.py) contains single seed analysis
        └── paper_figures/              # Paper figures created by collocation_figures-jibs.R

## Expected output

The main script creates the simulation folder and its sub-folders. In the simulation folder each seed receives an own folder (with the respective seed number). The final merged results (merged_data.csv) for all seeds are stored in results. Make sure that the working directory is set to the folder containing all scripts. Note of caution: System-related factors (e.g., different CPU microarchitectures) can produce deviating results. 

The single seed analysis is stored into a shap folder within results.
