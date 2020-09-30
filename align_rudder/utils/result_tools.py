from glob import glob
import os
import json
import numpy as np
import pandas as pd
from natsort import natsorted
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

sns.set(rc={"text.usetex": False})


def collect_run_results(run_dir, method, log_type="npy"):
    if log_type == "npy":
        results_files = glob(os.path.join(run_dir, "*", "num_episodes_*.npy"))

    result_dict = {method: {}}
    for logfile in results_files:
        with open(os.path.join(os.path.dirname(logfile), "params.json"), "r") as pf:
            params = json.load(pf)
        # read params
        num_demos = params["num_demo_use"]
        env_name = params["env_id"]
        if "FourRooms" in env_name:
            env_name = "FourRooms"
        elif "EightRooms" in env_name:
            env_name = "EightRooms"
        else:
            continue
        # prepare dict
        if env_name not in result_dict[method]:
            result_dict[method][env_name] = {}
        if num_demos not in result_dict[method][env_name]:
            result_dict[method][env_name][num_demos] = []
        # record
        if log_type == "csv":
            data = pd.read_csv(logfile)
            num_eps = int(data["iteration"].iloc[0]) + 1
        elif log_type == "npy":
            num_eps = int(np.load(logfile))
        result_dict[method][env_name][num_demos].append(num_eps)
    return result_dict


def merge_dicts(source, destination):
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            merge_dicts(value, node)
        else:
            destination[key] = value
    return destination


def collect_results(results_dir="results"):
    run_params = [
        {"run_dir": os.path.join(results_dir, "eight_rooms_alignrudder"), "method": "Align-RUDDER", "log_type": "npy"},
        {"run_dir": os.path.join(results_dir, "four_rooms_alignrudder"), "method": "Align-RUDDER", "log_type": "npy"},
        {"run_dir": os.path.join(results_dir, "four_rooms_bc"), "method": "BC + Q-Learning", "log_type": "npy"},
        {"run_dir": os.path.join(results_dir, "eight_rooms_bc"), "method": "BC + Q-Learning", "log_type": "npy"},
        {"run_dir": os.path.join(results_dir, "eight_rooms_dqfd"), "method": "DQfD", "log_type": "npy"},
        {"run_dir": os.path.join(results_dir, "four_rooms_dqfd"), "method": "DQfD", "log_type": "npy"}]

    result_dict = {}
    for params in run_params:
        if os.path.exists(params["run_dir"]):
            merge_dicts(collect_run_results(**params), result_dict)

    return result_dict


def print_summary(results_dir="results", env_name=None):
    results = collect_results(results_dir=results_dir)
    for method in natsorted(results.keys()):
        method_results = results[method]
        print(f"=== {method} ===")
        for env, data in method_results.items():
            print(f"Environment: {env}")
            for num_demos in natsorted(data.keys()):
                episodes = data[num_demos]
                print(f" N={num_demos}: {np.mean(episodes)}")


def plot_results(results_dir="results", env_name="FourRooms", outfile="results_four_rooms.pdf"):
    result_list = []
    results = collect_results(results_dir=results_dir)
    for method in natsorted(results.keys()):
        method_results = results[method]
        for env, data in method_results.items():
            if env == env_name:
                for num_demos in natsorted(data.keys()):
                    episodes = data[num_demos]
                    for e in episodes:
                        result_list.append([method, num_demos, e])

    if len(result_list) > 0:
        df = pd.DataFrame(result_list)
        df.columns = ["Method", "Demonstrations", "Episodes"]
        df = df.groupby(['Method', 'Demonstrations'])['Episodes'].nsmallest(100).reset_index()
        print(df.Method.value_counts())
        colors = sns.color_palette('deep')
        sns.set_style("darkgrid")
        sns.set_context(rc={"lines.linewidth": 0.75})
        rc('font', **{'family': 'serif', 'serif': ['DejaVu Sans']})
        fig, ax = plt.subplots(figsize=(6, 3))
        g = sns.pointplot(data=df, x="Demonstrations", y="Episodes", hue="Method",
                          capsize=.05, palette=colors, estimator=np.mean, ci=90, ax=ax)
        ax.set_xlabel(f"Demonstrations of the {env_name} Environment")
        ax.set_ylabel(r"Episodes to 80\% optimal return")
        ax.legend().set_title(None)

        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight")
            print(f"Plot saved in {outfile}")
            df.to_pickle(results_dir + '/' + env_name)  # where to save it, usually as a .pkl
