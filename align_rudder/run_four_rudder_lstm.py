import os
import pkg_resources
import numpy as np
import random
from ray import tune
import ray
import gym
from align_rudder.learning.rudder import RudderLearn
import shutil

config = {
    'env_id': 'align_rudder:FourRooms-v0',  # environment for the experiment
    'num_demo_use': tune.grid_search([2, 5, 10, 50, 100]),  # number of demonstrations
    'seed': tune.grid_search([i for i in range(10)]),  # Seed for experiment
    'demo_path': 'demonstrations/four_rooms.npy'
}


def run(config):
    run_path = os.getcwd()
    env_id = config['env_id']
    env = gym.make(env_id)
    demo_path = pkg_resources.resource_filename("align_rudder", config["demo_path"])
    # set seed
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    rl = RudderLearn(env=env, demo_path=demo_path, num_demo_use=config['num_demo_use'], run_path=run_path)

    rl.learn()


if __name__ == "__main__":
    if os.path.exists(os.path.join("results", "four_rooms_rudderlstm")):
        shutil.rmtree(os.path.join("results", "four_rooms_rudderlstm"))

    ray.init(temp_dir='/tmp/ray-four-lstm', log_to_driver=False)
    print("Starting Runs...")
    tune.run(run, config=config, local_dir="results", name="four_rooms_rudderlstm",
             resources_per_trial={'cpu': 2}, queue_trials=True)
    print("Finished!")
