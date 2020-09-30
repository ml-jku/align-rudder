import ray
from ray import tune
import gym
from align_rudder.learning.q_learning import Qlearning
import numpy as np
import random
import os
import pkg_resources
import shutil

config = {
    'env_id': 'align_rudder:EightRooms-v0',  # environment for the experiment
    'exp_name': 'align-rudder',  # name of the experiment
    'gamma': 1.0,  # Discount factor for q learning algorithm
    'total_timesteps': 10000000,
    'max_episodes': 100000,
    'learning_rate': 0.1,
    'epsilon': 0.2,  # exploration constant
    'num_seq_store': 10,  # max sequences to use for alignment or storing
    'num_clusters': 15,  # Number of clusters to use in k-means
    'consensus_thresh': 0.9,  # Threshold for consensus
    'eval': 40,
    'top_n': 12,
    'rudder': True,  # Use rudder or not
    'mode': 'log',  # 'log' or 'exp'
    'stop_criteria': '80opt',
    'enough_seq': 3,  # How many sequences are enough for sequence alignment
    'num_demo_use': tune.grid_search([2, 5, 10, 50, 100]),  # number of demonstrations
    'consensus_type': 'all',  # Select between most common or threshold all sequences: all, most_common
    'cluster_type': 'AP',  # Use default clustering, SpectralClustering, AffinityPropogation: default, SC, AP
    'seed': tune.grid_search([i for i in range(10)]),  # Seed for experiment
    'anneal_eps': 1.0,  # annealing rate for exploration
    'eps_lb': 0.0,  # eps anneal lower bound
    'rr_thresh': 0.005,  # Inverse visitation freq below thresh, set rr to zero
    'log_every': 10,  # log every timesteps
    'normalise_rr_by_max': True,  # normalize rr by maximum reward in rr
    'normalisation_scale': 10,  # scale factor compared to original reward
    'use_succ': True,
    'use_demo': True,
    'demo_path': 'demonstrations/eight_rooms.npy',
    'update_alignment:': False,
    'max_reward': 1,
    'use_exp_replay': False,
    'memory_len': 30000,
    'init_mean': False,
    'use_new_form': True
}


def run(config):
    run_path = os.getcwd()
    env_id = config['env_id']
    env = gym.make(env_id)
    # set seed
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    demo_path = pkg_resources.resource_filename("align_rudder", config["demo_path"])

    rl = Qlearning(env=env, eps=config['epsilon'], alpha=config['learning_rate'],
                   total_timesteps=config['total_timesteps'],
                   num_store_seq=config['num_seq_store'], rudder=config['rudder'], enough_seq=config['enough_seq'],
                   num_clusters=config['num_clusters'], top_n=config['top_n'],
                   consensus_type=config['consensus_type'],
                   consensus_thresh=config['consensus_thresh'], cluster_type=config['cluster_type'],
                   run_path=run_path,
                   anneal_eps=config['anneal_eps'], eps_lb=config['eps_lb'], rr_thresh=config['rr_thresh'],
                   log_every=config['log_every'], normalise_rr_by_max=config['normalise_rr_by_max'],
                   normalisation_scale=config['normalisation_scale'], use_succ=config['use_succ'],
                   use_demo=config['use_demo'],
                   demo_path=demo_path,
                   num_demo_use=config['num_demo_use'],
                   max_episodes=config['max_episodes'], max_reward=config['max_reward'],
                   mode=config['mode'],
                   gamma=config['gamma'], stop_criteria=config['stop_criteria'], seed=config['seed'],
                   init_mean=config['init_mean'], use_new_form=config['use_new_form'])

    rl.learn()


if __name__ == "__main__":
    # clear output dir
    if os.path.exists(os.path.join("results", "eight_rooms_alignrudder")):
        shutil.rmtree(os.path.join("results", "eight_rooms_alignrudder"))

    ray.init(temp_dir='/tmp/ray-eight-align', log_to_driver=False)
    print("Starting Runs...")
    # run(config)
    tune.run(run, config=config, local_dir="results/", name="eight_rooms_alignrudder")
    print("Finished!")
