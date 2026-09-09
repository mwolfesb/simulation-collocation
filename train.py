import os
import itertools
import copy

from shutil import rmtree

import torch
import torch.optim as optim

from firm_world import FirmWorldEnv
from ppo_cnn_agent import CNNAgent
from clean_ppo_selfplay import CleanPPOSelfPlay
from future_profit_max import FutureProfitMax
from merge_csv import merge_csv_files


## Training hyperparameters
NUM_ROUNDS = 30         # Round-robin: number of training rounds
TRAINING_PERIOD = 1000  # Number of training steps per round (interval at which trained and fixed agent switch)
UPDATE_EPOCHS = 3       # PPO-specific hyperparameter: defines how often the policy is updated using the same experience data (improves learning; more epochs could result in overfitting)
EVAL_FREQ = 100         # Once every EVAL_FREQ episodes of training, evaluation of policy quality is happening (should stay fixed)

## PPO - Hyperparameters
ENT_COEF = 0.01         # Entropy coefficient: encourages exploration of the agent (0.01 is common starting point; can be increased if agent becomes too deterministic too fast)
ENT_DECAY = 0.998       # Entropy decay (regularizes the entropy bonus; more exploratory in the beginning and more deterministic later; 0.995-0.999 are common values with higher values leading to prolonged exploration)
VF_COEF = 0.1           # Value function coefficient: balances the policy loss (actor) vs. value loss (critic value) 0.1 prioritizes optimizing policy loss
CLIP_COEF = 0.1         # Clip coefficient, makes sure that agent does not drift too far from where he is (common values are 0.1-0.3; lower values are more conservative)
GAMMA = 0.99            # Discount factor from Belman-equation (time-value of future rewards; usually fixed at this value)
GAE_LAMBDA = 0.99       # Generalized advantage estimate: allows agent to learn, taking into account both immediate, as well as future rewards (0.99 common value paired with GAMMA 0.99)
BATCH_SIZE = 32         # Number of samples used in one update (with GPU increase to 64; would make it faster)

PATH = 'simulation'                 # Path for storing results and trained models
# SEEDS = list(range(1, 10_001))      # Set the number of seeds (10_000; parallel computation speeds up computation)
SEEDS = list(range(1, 3))      # Set the number of seeds (10_000; parallel computation speeds up computation)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)        #Added for improved performance on university cluster server

    # Remove existing results directory
    if os.path.exists(PATH):
        rmtree(PATH)
    
    # Create master plot and results directorates
    os.makedirs(PATH + '/learning_curves')
    os.makedirs(PATH + '/results')

    # Iterate over all seeds and number of agents, and train combinations
    for seed, num_firms in itertools.product(SEEDS, [2]):
        print(f'Training with seed: {seed}')

        # Create a directory for storing training model and results
        train_path = (PATH + '/' + str(seed) + '/')
        # Create training path + folders
        os.makedirs(train_path)
        os.makedirs(train_path + 'results')
        os.makedirs(train_path + 'models')

        # Evaluate a fixed policy and save the results
        fixed_env = FirmWorldEnv(
            num_countries=50, 
            num_years=50,
            seed=seed,  
            num_firms=num_firms,
            comp_int_value = [0,1],
            FMA = [1,16], 
            entry_barriers = [1000,7000],
            country_spread = [100,2000],
            LOF = [0.5,5],
            LOF_reduction = [0,0.25],
            alpha = [0.8,1.2],
            beta = [0.7,1],
            path=train_path, 
            save_results=True)

        # Create a fixed policy - Future Profit Maximizing Policy
        fixed_policy = FutureProfitMax(fixed_env)

        # Evaluate the agents with the fixed policy
        obs, info = fixed_env.reset()
        _, _ = fixed_policy.simulate(obs)

        # Save the results and close the environment
        fixed_env.close()

        # Set the environments (for training)
        train_env = FirmWorldEnv(
            num_countries=50, 
            num_years=50,
            seed=seed,  
            num_firms=num_firms,
            comp_int_value = [0,1],
            FMA = [1,16], 
            entry_barriers = [1000,7000],
            country_spread = [100,2000],
            LOF = [0.5,5],
            LOF_reduction = [0,0.25],
            alpha = [0.8,1.2],
            beta = [0.7,1],
            path=train_path,
            save_results=True)
        
        #returns the possible actions of the (first) agent
        num_actions = train_env.action_space(train_env.possible_agents[0]).n
        #returns the possible observation space (as initial layer for the neural network)
        obs_depth = train_env.observation_space(train_env.possible_agents[0]).shape[0]

        # Set the environments (for evaluation)
        eval_env = FirmWorldEnv(
            num_countries=50, 
            num_years=50,
            seed=seed,  
            num_firms=num_firms,
            comp_int_value = [0,1],
            FMA = [1,16], 
            entry_barriers = [1000,7000],
            country_spread = [100,2000],
            LOF = [0.5,5],
            LOF_reduction = [0,0.25],
            alpha = [0.8,1.2],
            beta = [0.7,1],
            path=train_path,
            save_results=False)

        # Set random-gen seed for PyTorch models (neural network library)
        torch.manual_seed(seed)

        # RL Agents - Policy and Value Neural Networks for each RL agent (firm)
        agents = {'firm_0': CNNAgent(
            num_countries=50, 
            num_actions=num_actions,
            obs_depth=obs_depth, 
            l1_kernels=8, l2_kernels=8,
            fc_size=200).to(device)}
        
        # Copy the first agent's policy and value neural networks for second agent
        agents['firm_1'] = copy.deepcopy(agents['firm_0'])

        # Optimizer - Adam - One for each agent
        optimizers = {a: optim.Adam(agents[a].parameters(), lr=0.0005, eps=1e-5)
                      for a in agents}

        # Entropy Coefficient per agent - for exploration - One for each agent
        ent_coef = {a: ENT_COEF for a in train_env.possible_agents}
        num_episodes = ((NUM_ROUNDS * TRAINING_PERIOD * num_firms) + TRAINING_PERIOD)

        # PPO algorithm with SelfPlay MARL
        ppo = CleanPPOSelfPlay(
            train_env=train_env, eval_env=eval_env, 
            agents=agents, optimizers=optimizers, 
            device=device)

        # Train the agents using SelfPlay
        ppo.train(ent_coef=ent_coef, 
                  ent_decay=ENT_DECAY, 
                  vf_coef=VF_COEF,
                  gamma=GAMMA, 
                  clip_coef=CLIP_COEF, 
                  batch_size=BATCH_SIZE,
                  gae_lambda=GAE_LAMBDA, 
                  num_episodes=num_episodes,
                  update_epochs=UPDATE_EPOCHS,
                  training_period=TRAINING_PERIOD,
                  eval_freq=EVAL_FREQ,
                  path=train_path)
        
        # Close all the simulation environments
        train_env.close()
        eval_env.close()
    
    # Merge all seeds and final processing of results
    merge_csv_files(root_dir = PATH, 
                    output_dir=(PATH + '/results'), 
                    output_filename='merged_data.csv', 
                    seeds=SEEDS)

if __name__ == "__main__":
    main()