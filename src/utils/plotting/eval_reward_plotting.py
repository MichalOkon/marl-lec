import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.utils.plotting.helper_functions import get_ci_bootstrap

DEFAULT_XLABEL = 'PV Generation Coefficient'
DEFAULT_YLABEL = 'Average Cost per Timestep [¢]'
DEFAULT_TITLE = 'Final Evaluation Costs for Different PV Generation Values'
COMMON_BATTERY_TITLE = 'Final Evaluation Costs for Different Common Battery Reward Coefficients'
COMMON_BATTERY_XLABEL = 'Common Battery Reward Coefficient'


def average_final_evaluation_rewards(dfs):
    if not dfs:
        raise ValueError("No data to average.")
    # Flatten rewards from all data dictionaries.
    rewards = [r for data in dfs for r in data['best_evaluation_rewards_per_ts_per_hh'].values()]
    averaged_reward = np.mean(rewards)
    error_reward = get_ci_bootstrap(rewards)
    return averaged_reward, error_reward


def average_final_evaluation_rewards_from_root_dir(root_dir):
    dataframes = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == "results.json":
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    dataframes.append(data)
    averaged_reward, error_reward = average_final_evaluation_rewards(dataframes)
    baseline_reward = dataframes[0]['baseline_test_reward_per_timestep']
    return averaged_reward, error_reward, baseline_reward


def average_final_evaluation_rewards_from_dirs(directories):
    dataframes = []
    for directory in directories:
        results_file = os.path.join(directory, 'results.json')
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
                dataframes.append(data)
        except FileNotFoundError:
            print(f"File {results_file} not found.")
        except json.JSONDecodeError:
            print(f"Error reading file {results_file}.")
    averaged_reward, error_reward = average_final_evaluation_rewards(dataframes)
    baseline_reward = dataframes[0]['baseline_test_reward_per_timestep']
    return averaged_reward, error_reward, baseline_reward


# === Plotting Functions ===

def plot_multiple_final_eval_cost(rewards, errors, titles, baseline_rewards=None, save_path=None,
                                  alt_xlabel=None, alt_ylabel=None, alt_title=None, limit_y=False):
    """
    Create a bar plot of average final evaluation costs with error bars.

    Parameters:
      - rewards: list of lists (or arrays) of rewards for each file, in the same order as titles.
      - errors: corresponding error values (e.g. confidence intervals).
      - titles: labels for the x-axis.
      - baseline_rewards: if provided, draw dashed baseline lines on the bars.
      - save_path: if provided, save the figure to the given path.
      - alt_xlabel, alt_ylabel, alt_title: alternative axis labels and title.
      - limit_y: if True, adjust the y-axis limits based on the data.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    global_min = float('inf')
    global_max = float('-inf')

    for i, reward_values in enumerate(rewards):
        err = errors[i]
        mean_reward = -np.mean(reward_values)
        ax.bar(i, mean_reward, yerr=err, error_kw=dict(ecolor='black', elinewidth=1, capsize=3))
        global_min = min(global_min, mean_reward - err)
        global_max = max(global_max, mean_reward + err)

    if baseline_rewards is not None:
        for i, baseline in enumerate(baseline_rewards):
            bar_width = ax.patches[i].get_width()
            bar_x = ax.patches[i].get_x()
            ax.hlines(y=-baseline, xmin=bar_x, xmax=bar_x + bar_width, color='purple', linestyle='--')

    custom_line = Line2D([0], [0], color='purple', linestyle='--', lw=2)
    ax.legend([custom_line], ['Baseline value'], loc='upper left')

    ax.set_xticks(range(len(titles)))
    ax.set_xticklabels(titles)
    ax.set_xlabel(alt_xlabel if alt_xlabel is not None else DEFAULT_XLABEL)
    ax.set_ylabel(alt_ylabel if alt_ylabel is not None else DEFAULT_YLABEL)
    ax.set_title(alt_title if alt_title is not None else DEFAULT_TITLE)

    if limit_y:
        margin = (global_max - global_min) * 0.1
        y_min = global_min - margin if global_min != float('inf') else 0
        y_max = global_max + margin if global_max != float('-inf') else 1
        if baseline_rewards is not None:
            y_min = min(y_min, -max(baseline_rewards) - margin)
            y_max = max(y_max, -min(baseline_rewards) + margin)
        ax.set_ylim(y_min, y_max)

    if save_path is not None:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)


def read_and_plot_multiple_final_eval_rewards(result_dirs, titles, save_path=None,
                                              common_battery_reward_mode=False, limit_y=False):
    """
    For each directory in result_dirs, load the results.json file, compute the averaged reward,
    and plot them together.
    """
    baseline_rewards = []
    rewards = []
    rewards_errors = []
    for directory in result_dirs:
        avg_reward, error_reward, baseline_reward = average_final_evaluation_rewards_from_root_dir(directory)
        rewards.append(avg_reward)
        rewards_errors.append(error_reward)
        baseline_rewards.append(baseline_reward)

    if common_battery_reward_mode:
        plot_multiple_final_eval_cost(
            rewards, rewards_errors, titles, baseline_rewards, save_path,
            alt_title=COMMON_BATTERY_TITLE,
            alt_xlabel=COMMON_BATTERY_XLABEL,
            limit_y=limit_y
        )
    else:
        plot_multiple_final_eval_cost(
            rewards, rewards_errors, titles, baseline_rewards, save_path, limit_y=limit_y
        )


def read_and_plot_multiple_final_eval_rewards_grouped(result_dir, groups, titles, save_path=None,
                                                      limit_y=False, alt_title=None, alt_xlabel=None,
                                                      common_battery_reward_mode=True):
    """
    Group directories by prefixes in groups, average the rewards for each group, and plot them.
    """
    baseline_rewards = []
    rewards = []
    rewards_errors = []
    print("Result dir:", result_dir)

    for group in groups:
        result_directories = []
        for root, dirs, files in os.walk(result_dir):
            for d in dirs:
                if d.startswith(group):
                    result_directories.append(os.path.join(root, d))
        print(f"Group '{group}' found directories: {result_directories}")
        avg_reward, error_reward, baseline_reward = average_final_evaluation_rewards_from_dirs(result_directories)
        rewards.append(avg_reward)
        rewards_errors.append(error_reward)
        baseline_rewards.append(baseline_reward)

    if common_battery_reward_mode:
        plot_multiple_final_eval_cost(
            rewards, rewards_errors, titles, baseline_rewards, save_path,
            alt_title=COMMON_BATTERY_TITLE,
            alt_xlabel=COMMON_BATTERY_XLABEL,
            limit_y=limit_y
        )
    else:
        plot_multiple_final_eval_cost(
            rewards, rewards_errors, titles, baseline_rewards, save_path,
            alt_title=alt_title, alt_xlabel=alt_xlabel, limit_y=limit_y
        )
