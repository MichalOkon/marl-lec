import glob
import json

import random
import re
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os
import pandas as pd
from pyomo.contrib.parmest.graphics import sns
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
import matplotlib.lines as mlines

def calculate_and_plot_match_outcomes(experiment_id, match_results_filename=None, plot_filename="match_results_plot"):
    if match_results_filename is None:
        match_results_filename = "match_results"
    with open('results/{:s}/{:s}.json'.format(experiment_id, match_results_filename)) as file:
        match_results = json.load(file)

    chicken = []
    prisoners = []
    stag_hunt = []
    non_ssd_r_greater_p = []
    non_ssd_r_smaller_p = []

    for single_result in match_results:
        results_dict = single_result["results"]
        household_id1, household_id2 = list(results_dict["abundant"]['env_runners']['policy_reward_mean'].keys())
        R = results_dict["abundant"]['env_runners']["episode_reward_mean"] / 2
        P = results_dict["scarce"]['env_runners']["episode_reward_mean"] / 2
        S = (results_dict["mixed1"]['env_runners']["policy_reward_mean"][household_id1] +
             results_dict["mixed2"]['env_runners']["policy_reward_mean"][
                 household_id2]) / 2
        T = (results_dict["mixed1"]['env_runners']["policy_reward_mean"][household_id2] +
             results_dict["mixed2"]['env_runners']["policy_reward_mean"][
                 household_id1]) / 2

        fear = P - S
        greed = T - R

        is_non_ssd = False
        if R <= P:
            print("Game is non-SSD because R <= P")
            is_non_ssd = True
        if R <= S:
            print("Game is non-SSD because R <= S")
            is_non_ssd = True
        if 2 * R <= T + S:
            print("Game is non-SSD because 2R <= T + S")
            is_non_ssd = True
        if T <= R and P <= S:
            print("Game is non-SSD because T <= R and P <= S")
            is_non_ssd = True
        if is_non_ssd:
            if R > P:
                non_ssd_r_greater_p.append((fear, greed))
            else:
                non_ssd_r_smaller_p.append((fear, greed))
        else:
            if fear > 0 and greed > 0:
                prisoners.append((fear, greed))
            elif fear > 0 and greed < 0:
                stag_hunt.append((fear, greed))
            elif fear < 0 and greed > 0:
                chicken.append((fear, greed))

    chicken = np.array(chicken)
    prisoners = np.array(prisoners)
    stag_hunt = np.array(stag_hunt)
    non_ssd_r_greater_p = np.array(non_ssd_r_greater_p)
    non_ssd_r_smaller_p = np.array(non_ssd_r_smaller_p)

    fig, ax = plt.subplots(figsize=(10, 6))

    margin = 20
    x_min = min(
        [array[:, 0].min() for array in [chicken, prisoners, stag_hunt, non_ssd_r_smaller_p, non_ssd_r_greater_p] if
         len(array) > 0]) - margin
    x_max = max(
        [array[:, 0].max() for array in [chicken, prisoners, stag_hunt, non_ssd_r_smaller_p, non_ssd_r_greater_p] if
         len(array) > 0]) + margin

    y_min = min(
        [array[:, 1].min() for array in [chicken, prisoners, stag_hunt, non_ssd_r_smaller_p, non_ssd_r_greater_p] if
         len(array) > 0]) - margin
    y_max = max(
        [array[:, 1].max() for array in [chicken, prisoners, stag_hunt, non_ssd_r_smaller_p, non_ssd_r_greater_p] if
         len(array) > 0]) + margin

    max_val_x = max(abs(x_min), abs(x_max))
    max_val_y = max(abs(y_min), abs(y_max))
    max_val = max(max_val_x, max_val_y)
    ax.fill_between(x=[0, 1], y1=0, y2=1, color='red', alpha=0.2)
    ax.fill_between(x=[-1, 0], y1=0, y2=1, color='blue', alpha=0.2)
    ax.fill_between(x=[-1, 0], y1=-1, y2=0, color='grey', alpha=0.2)
    ax.fill_between(x=[0, 1], y1=-1, y2=0, color='green', alpha=0.2)

    if len(chicken) > 0:
        ax.scatter(chicken[:, 0] / max_val, chicken[:, 1] / max_val, c='blue', marker='s', label='Chicken',
                   alpha=0.8)
    if len(prisoners) > 0:
        ax.scatter(prisoners[:, 0] / max_val, prisoners[:, 1] / max_val, c='red', marker='D',
                   label='Prisoner\'s dilemma', alpha=0.6)
    if len(stag_hunt) > 0:
        ax.scatter(stag_hunt[:, 0] / max_val, stag_hunt[:, 1] / max_val, c='green', marker='o',
                   label='Stag hunt', alpha=0.8)
    if len(non_ssd_r_smaller_p) > 0:
        ax.scatter(non_ssd_r_smaller_p[:, 0] / max_val, non_ssd_r_smaller_p[:, 1] / max_val, c='black',
                   marker='*',
                   label='Non-SSD (R < P)',
                   alpha=0.6)
    if len(non_ssd_r_greater_p) > 0:
        ax.scatter(non_ssd_r_greater_p[:, 0] / max_val, non_ssd_r_greater_p[:, 1] / max_val, c='black',
                   marker='x', edgecolor='grey',
                   label="Non-SSD (R > P)", alpha=0.4)

    ax.axvline(0, color='black', linewidth=1)  # Vertical line for quadrant separation
    ax.axhline(0, color='black', linewidth=1)  # Horizontal line for quadrant separation
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Fear')
    ax.set_ylabel('Greed')
    ax.set_title('Classification of Social Dilemmas by Quadrant')

    ax.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
    plt.savefig('results/{:s}/{:s}.png'.format(experiment_id, plot_filename))


def aggregate_match_runs(parent_dirs, output_prefixes, time_threshold_minutes=2):
    """
    Each run outputs several evaluations round results, which are stored in a directory with a timestamp.
    This function aggregates the results coming from the same match run, and saves them in a single file.
    """
    def parse_timestamp(dir_name):
        try:
            timestamp_str = dir_name.split('_')[-2:]
            return datetime.strptime('_'.join(timestamp_str), "%Y-%m-%d_%H-%M-%S")
        except Exception as e:
            print(f"Error parsing timestamp for {dir_name}: {e}")
            return None

    for parent_dir, output_prefix in zip(parent_dirs, output_prefixes):
        directories = []
        for item in os.listdir(parent_dir):
            dir_path = os.path.join(parent_dir, item)
            if os.path.isdir(dir_path) and item.startswith("run_results"):
                timestamp = parse_timestamp(item)
                if timestamp:
                    directories.append((timestamp, dir_path))
        directories.sort(key=lambda x: x[0])
        groups = []
        current_group = [directories[0]] if directories else []
        for i in range(1, len(directories)):
            time_diff = directories[i][0] - current_group[-1][0]
            if time_diff <= timedelta(minutes=time_threshold_minutes):
                current_group.append(directories[i])
            else:
                groups.append(current_group)
                current_group = [directories[i]]
        if current_group:
            groups.append(current_group)
        for group_index, group in enumerate(groups):
            group_data = []
            for _, dir_path in group:
                file_path = os.path.join(dir_path, "accumulated_household_logs.csv")
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        group_data.append(df)
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
            if group_data:
                combined_df = pd.concat(group_data, ignore_index=True)
                combined_df.columns.values[0] = "household_id"
                averaged_df = combined_df.groupby("household_id").mean().reset_index()
                output_file = f"{output_prefix}{group_index + 1}.csv"
                averaged_df.to_csv(output_file, index=False)
                print(f"Group {group_index + 1} aggregated and saved to {output_file}")
            else:
                print(f"No valid data found for group {group_index + 1}")

def analyze_battery_data(data_dir):
    """
    Compare the battery charge and discharge data of two households. Perform statystical tests to determine if the
    data is normally distributed and if the means are significantly different.
    """

    data_frames = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path)
            data_frames.append(df)

    data = pd.concat(data_frames, ignore_index=True)

    house_1 = data[data['household_id'] == '27_0']
    house_2 = data[data['household_id'] == '27_1']

    charge_1 = house_1['total_common_battery_charge']
    charge_2 = house_2['total_common_battery_charge']

    discharge_1 = house_1['total_common_battery_discharge']
    discharge_2 = house_2['total_common_battery_discharge']

    charge_1_normality = shapiro(charge_1)
    charge_2_normality = shapiro(charge_2)
    discharge_1_normality = shapiro(discharge_1)
    discharge_2_normality = shapiro(discharge_2)

    print("Normality Test Results:")
    print(f"Charge 1: Statistic={charge_1_normality.statistic}, p-value={charge_1_normality.pvalue}")
    print(f"Charge 2: Statistic={charge_2_normality.statistic}, p-value={charge_2_normality.pvalue}")
    print(f"Discharge 1: Statistic={discharge_1_normality.statistic}, p-value={discharge_1_normality.pvalue}")
    print(f"Discharge 2: Statistic={discharge_2_normality.statistic}, p-value={discharge_2_normality.pvalue}")

    if all(p > 0.05 for p in [charge_1_normality.pvalue, charge_2_normality.pvalue]):
        print("\nBattery charge data is normally distributed.")
        charge_t_stat, charge_p_val = ttest_ind(charge_1, charge_2, alternative='less')
    else:
        print("\nBattery charge data is not normally distributed.")
        charge_u_stat, charge_u_p_val = mannwhitneyu(charge_1, charge_2, alternative='less')

    if all(p > 0.05 for p in [discharge_1_normality.pvalue, discharge_2_normality.pvalue]):
        print("Battery discharge data is normally distributed.")
        discharge_t_stat, discharge_p_val = ttest_ind(discharge_1, discharge_2, alternative='greater')
    else:
        print("Battery discharge data is not normally distributed.")
        discharge_u_stat, discharge_u_p_val = mannwhitneyu(discharge_1, discharge_2, alternative='greater')

    print("\nTest Results:")
    if 'charge_t_stat' in locals():
        print(f"T-test for charge: Statistic={charge_t_stat}, p-value={charge_p_val}")
    else:
        print(f"Mann-Whitney U test for charge: Statistic={charge_u_stat}, p-value={charge_u_p_val}")

    if 'discharge_t_stat' in locals():
        print(f"T-test for discharge: Statistic={discharge_t_stat}, p-value={discharge_p_val}")
    else:
        print(f"Mann-Whitney U test for discharge: Statistic={discharge_u_stat}, p-value={discharge_u_p_val}")

def get_results_by_coeff(root_dir):
    """
    Searches through the directory tree starting at root_dir for JSON files named "results.json",
    extracts the coefficient from their parent folder name (using a regex), and computes the averaged
    best evaluation rewards (grouped by coefficient).
    """
    results_by_coeff = {}

    json_files = glob.glob(os.path.join(root_dir, '**', 'results.json'), recursive=True)

    # Expected folder names: "dilemma_0_run_results_01" or "dilemma_2_5_run_results_01", etc.
    pattern = re.compile(r'dilemma_([\d_]+)_run_results')

    for json_path in json_files:
        parent_dir = os.path.basename(os.path.dirname(json_path))
        match = pattern.search(parent_dir)
        if not match:
            continue

        coeff_str = match.group(1)
        try:
            coeff_val = float(coeff_str.replace('_', '.'))
        except ValueError:
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        best_rewards = data.get("best_evaluation_rewards_per_ts_per_hh")
        if not best_rewards:
            continue

        try:
            rewards = [float(r) for r in best_rewards.values()]
        except (TypeError, ValueError):
            continue

        avg_reward = np.mean(rewards)
        results_by_coeff.setdefault(str(coeff_val), []).append(avg_reward)

    results_by_coeff = OrderedDict(sorted(results_by_coeff.items(), key=lambda item: float(item[0])))
    return results_by_coeff

def plot_and_save_rewards(results_by_coeff, plot_type='boxplot', file_name="plot_rewards.png", dpi=300,
                          title=None, xlabel=None, ylabel=None, baseline=None, sort_numerically=False):
    """
    Plots averaged rewards grouped by coefficient
    """
    data_list = []
    for coeff, values in results_by_coeff.items():
        for val in values:
            data_list.append({'Coefficient': coeff, 'Avg Reward': val})
    df = pd.DataFrame(data_list)

    if sort_numerically:
        df['Coefficient'] = pd.to_numeric(df['Coefficient'], errors='coerce')
        df = df.sort_values('Coefficient')
        order = sorted(df['Coefficient'].unique(), key=float)

        order = [str(x) for x in order]

        df['Coefficient'] = df['Coefficient'].astype(str)

        if baseline is not None and isinstance(baseline, dict):
            baseline = {str(k): v for k, v in baseline.items()}
    else:
        df['Coefficient'] = df['Coefficient'].astype(str)
        order = sorted(df['Coefficient'].unique())

    palette_colors = sns.color_palette("Set1", n_colors=len(order))
    color_map = dict(zip(order, palette_colors))

    plt.figure(figsize=(10, 6))

    if plot_type.lower() == 'boxplot':
        sns.boxplot(x='Coefficient', y='Avg Reward', data=df, palette=color_map, order=order)
        sns.swarmplot(x='Coefficient', y='Avg Reward', data=df, color='black', alpha=0.7, order=order)
        if title is None:
            title = "Distribution of Averaged Best Evaluation Rewards per Coefficient"
        if xlabel is None:
            xlabel = "Coefficient"
        if ylabel is None:
            ylabel = "Average Best Evaluation Reward (per ts)"
    elif plot_type.lower() == 'violin':
        sns.violinplot(x='Coefficient', y='Avg Reward', data=df, inner='quartile', palette=color_map, order=order)
        sns.swarmplot(x='Coefficient', y='Avg Reward', data=df, color='black', alpha=0.7, order=order)
        if title is None:
            title = "Violin Plot of Averaged Best Evaluation Rewards per Coefficient"
        if xlabel is None:
            xlabel = "Coefficient"
        if ylabel is None:
            ylabel = "Average Best Evaluation Reward (per ts)"
    elif plot_type.lower() == 'point':
        sns.pointplot(x='Coefficient', y='Avg Reward', data=df, ci='sd', capsize=0.1, palette=color_map,
                      order=order)
        if title is None:
            title = "Point Plot with Standard Deviation Error Bars"
        if xlabel is None:
            xlabel = "Coefficient"
        if ylabel is None:
            ylabel = "Average Best Evaluation Reward (per ts)"
    else:
        raise ValueError("Invalid plot_type specified. Choose 'boxplot', 'violin', or 'point'.")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if baseline is not None:
        ax = plt.gca()
        if isinstance(baseline, dict):
            for idx, coeff in enumerate(order):
                if coeff in baseline:
                    plt.hlines(y=baseline[coeff], xmin=idx - 0.4, xmax=idx + 0.4,
                               colors='red', linestyles='dashed', linewidth=1.5)
            baseline_line = mlines.Line2D([], [], color='red', linestyle='--', label='Baseline')
            plt.legend(handles=[baseline_line])
        else:
            plt.axhline(y=baseline, color='red', linestyle='--', linewidth=1.5, label='Baseline')
            plt.legend()

    plt.savefig(file_name, dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close()

def perform_pairwise_analysis_baseline(results_by_coeff, baseline_coeff="0.0"):
    """
    Performs a pairwise comparison between the baseline group and all other groups.
    """
    baseline_key = str(baseline_coeff)
    if baseline_key not in results_by_coeff:
        raise ValueError(f"Baseline coefficient {baseline_key} not found in results_by_coeff")

    baseline_data = np.array(results_by_coeff[baseline_key])
    baseline_mean = np.mean(baseline_data)
    _, baseline_shapiro_p = shapiro(baseline_data)

    results = {}
    for coeff, group_vals in results_by_coeff.items():
        if str(coeff) == baseline_key:
            continue
        group_data = np.array(group_vals)
        group_mean = np.mean(group_data)
        diff = group_mean - baseline_mean
        percent_change = (diff / abs(baseline_mean) * 100) if baseline_mean != 0 else np.nan

        _, group_shapiro_p = shapiro(group_data)

        if baseline_shapiro_p > 0.05 and group_shapiro_p > 0.05:
            test_used = "t-test"
            test_stat, p_val = ttest_ind(baseline_data, group_data, equal_var=False)
        else:
            test_used = "Mann–Whitney U"
            test_stat, p_val = mannwhitneyu(baseline_data, group_data, alternative='two-sided')

        results[str(coeff)] = {
            "baseline_mean": baseline_mean,
            "group_mean": group_mean,
            "difference": diff,
            "percent_change": percent_change,
            "normality_baseline_p": baseline_shapiro_p,
            "normality_group_p": group_shapiro_p,
            "test_used": test_used,
            "test_statistic": test_stat,
            "p_value": p_val
        }

    for coeff, res in results.items():
        print(f"Comparison for coefficient {coeff}:")
        print(f"  Baseline Mean: {res['baseline_mean']:.4f}")
        print(f"  Group Mean: {res['group_mean']:.4f}")
        print(f"  Difference: {res['difference']:.4f}")
        print(f"  Percent Change: {res['percent_change']:.2f}%")
        print(
            f"  Normality p-values: Baseline = {res['normality_baseline_p']:.4f}, Group = {res['normality_group_p']:.4f}")
        print(
            f"  Test Used: {res['test_used']}, Test Statistic = {res['test_statistic']:.4f}, p-value = {res['p_value']:.4f}")
        print()

    return results


def plot_and_save_rewards_by_coeff(results_by_coeff, plot_type='boxplot', file_name="plot_rewards.png", dpi=300,
                          title=None, xlabel=None, ylabel=None, baseline=None, sort_numerically=False):
    """
    Plots averaged rewards grouped by the coefficient (or some other factor the rewards can be grouped by).
    """
    data_list = []
    for coeff, values in results_by_coeff.items():
        for val in values:
            data_list.append({'Coefficient': coeff, 'Avg Reward': val})
    df = pd.DataFrame(data_list)

    if sort_numerically:
        df['Coefficient'] = pd.to_numeric(df['Coefficient'], errors='coerce')
        df = df.sort_values('Coefficient')

        order = sorted(df['Coefficient'].unique(), key=float)
        order = [str(x) for x in order]

        df['Coefficient'] = df['Coefficient'].astype(str)

        if baseline is not None and isinstance(baseline, dict):
            baseline = {str(k): v for k, v in baseline.items()}
    else:
        df['Coefficient'] = df['Coefficient'].astype(str)
        order = sorted(df['Coefficient'].unique())

    palette_colors = sns.color_palette("Set1", n_colors=len(order))
    color_map = dict(zip(order, palette_colors))

    plt.figure(figsize=(10, 6))

    if plot_type.lower() == 'boxplot':
        sns.boxplot(x='Coefficient', y='Avg Reward', data=df, palette=color_map, order=order)
        sns.swarmplot(x='Coefficient', y='Avg Reward', data=df, color='black', alpha=0.7, order=order)
        if title is None:
            title = "Distribution of Averaged Best Evaluation Rewards per Coefficient"
        if xlabel is None:
            xlabel = "Coefficient"
        if ylabel is None:
            ylabel = "Average Best Evaluation Reward (per ts)"
    elif plot_type.lower() == 'violin':
        sns.violinplot(x='Coefficient', y='Avg Reward', data=df, inner='quartile', palette=color_map, order=order)
        sns.swarmplot(x='Coefficient', y='Avg Reward', data=df, color='black', alpha=0.7, order=order)
        if title is None:
            title = "Violin Plot of Averaged Best Evaluation Rewards per Coefficient"
        if xlabel is None:
            xlabel = "Coefficient"
        if ylabel is None:
            ylabel = "Average Best Evaluation Reward (per ts)"
    elif plot_type.lower() == 'point':
        sns.pointplot(x='Coefficient', y='Avg Reward', data=df, ci='sd', capsize=0.1, palette=color_map, order=order)
        if title is None:
            title = "Point Plot with Standard Deviation Error Bars"
        if xlabel is None:
            xlabel = "Coefficient"
        if ylabel is None:
            ylabel = "Average Best Evaluation Reward (per ts)"
    else:
        raise ValueError("Invalid plot_type specified. Choose 'boxplot', 'violin', or 'point'.")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if baseline is not None:
        ax = plt.gca()
        if isinstance(baseline, dict):
            for idx, coeff in enumerate(order):
                if coeff in baseline:
                    plt.hlines(y=baseline[coeff], xmin=idx - 0.4, xmax=idx + 0.4,
                               colors='red', linestyles='dashed', linewidth=1.5)
            baseline_line = mlines.Line2D([], [], color='red', linestyle='--', label='Baseline')
            plt.legend(handles=[baseline_line])
        else:
            plt.axhline(y=baseline, color='red', linestyle='--', linewidth=1.5, label='Baseline')
            plt.legend()

    plt.savefig(file_name, dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close()