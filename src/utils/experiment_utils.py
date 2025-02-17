import os
import glob
import json
from collections import OrderedDict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_1samp, wilcoxon


def get_results_and_baselines_by_label(root_dir, name_to_label_map):
    results_by_label = {}
    baselines_temp = {}

    json_files = glob.glob(os.path.join(root_dir, '**', 'results.json'), recursive=True)

    for json_path in json_files:
        parent_dir = os.path.basename(os.path.dirname(json_path))

        label = None
        for prefix, mapped_label in name_to_label_map.items():
            if parent_dir.startswith(prefix):
                label = mapped_label
                break
        if label is None:
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            continue

        best_rewards = data.get("best_evaluation_rewards_per_ts_per_hh")
        if not best_rewards:
            continue

        try:
            rewards = [float(r) for r in best_rewards.values()]
        except (TypeError, ValueError):
            continue

        avg_reward = np.mean(rewards)
        results_by_label.setdefault(label, []).append(avg_reward)

        baseline_val = data.get("baseline_test_reward_per_timestep")
        if baseline_val is not None:
            try:
                baseline_val = float(baseline_val)
            except (TypeError, ValueError):
                continue
            baselines_temp.setdefault(label, []).append(baseline_val)

    results_by_label = OrderedDict(sorted(results_by_label.items(), key=lambda item: item[0]))

    baselines_by_label = {}
    for label, baseline_list in baselines_temp.items():
        baselines_by_label[label] = np.mean(baseline_list)
    baselines_by_label = OrderedDict(sorted(baselines_by_label.items(), key=lambda item: item[0]))

    return results_by_label, baselines_by_label


def plot_and_save_rewards_by_label(results_by_label, plot_type='bar', file_name="plot_rewards.png", dpi=300,
                                   title=None, xlabel=None, ylabel=None, baseline=None, ci=95):
    """
    Plots averaged rewards grouped by a label. When a baseline is provided,
    the function computes and plots the % improvement from the baseline.
    """
    data_list = []
    for label, values in results_by_label.items():
        for val in values:
            data_list.append({'Label': label, 'Avg Reward': val})
    df = pd.DataFrame(data_list)

    df['Label'] = df['Label'].astype(str)

    if baseline is not None:
        if isinstance(baseline, dict):
            def compute_improvement(row):
                base = baseline.get(row['Label'])
                if base is None:
                    raise ValueError(f"No baseline provided for label {row['Label']}.")
                if base == 0:
                    raise ValueError(
                        f"Baseline value for label {row['Label']} is zero, cannot compute percentage improvement.")
                return (row['Avg Reward'] - base) / abs(base) * 100

            df['Improvement (%)'] = df.apply(compute_improvement, axis=1)
        else:
            if baseline == 0:
                raise ValueError("Baseline value is zero, cannot compute percentage improvement.")
            df['Improvement (%)'] = (df['Avg Reward'] - baseline) / baseline * 100

        y_column = 'Improvement (%)'
    else:
        y_column = 'Avg Reward'

    order = sorted(df['Label'].unique())

    palette_colors = sns.color_palette("Set1", n_colors=len(order))
    color_map = dict(zip(order, palette_colors))

    plt.figure(figsize=(10, 6))

    if plot_type.lower() == 'bar':
        sns.barplot(x='Label', y=y_column, data=df, palette=color_map, order=order, ci=ci)

        if baseline is not None:
            if title is None:
                title = "Bar Plot of % Improvement over Baseline by Label"
            if xlabel is None:
                xlabel = "Label"
            if ylabel is None:
                ylabel = "Improvement (%)"
        else:
            if title is None:
                title = "Bar Plot of Averaged Best Evaluation Rewards per Label"
            if xlabel is None:
                xlabel = "Label"
            if ylabel is None:
                ylabel = "Average Best Evaluation Reward (per ts)"
    else:
        raise ValueError("Invalid plot_type specified. For this function, please use 'bar'.")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if baseline is not None:
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Baseline (0% Improvement)')
        plt.legend()

    plt.savefig(file_name, dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close()

def perform_pairwise_analysis_by_coeff(results_by_coeff, baseline=None):
    results = {}

    for coeff, group_vals in results_by_coeff.items():
        coeff_str = str(coeff)

        if baseline is None or coeff_str not in baseline:
            raise ValueError(f"No baseline provided for coefficient {coeff_str}")

        baseline_value = baseline[coeff_str]
        group_data = np.array(group_vals)
        group_mean = np.mean(group_data)
        diff = group_mean - baseline_value
        percent_change = (diff / abs(baseline_value) * 100) if baseline_value != 0 else np.nan

        _, normality_group_p = shapiro(group_data)

        if normality_group_p > 0.05:
            test_used = "t-test"
            test_stat, p_val = ttest_1samp(group_data, baseline_value)
        else:
            test_used = "Wilcoxon"
            try:
                test_stat, p_val = wilcoxon(group_data - baseline_value)
            except Exception:
                test_stat, p_val = np.nan, np.nan

        results[coeff_str] = {
            "baseline": baseline_value,
            "group_mean": group_mean,
            "difference": diff,
            "percent_change": percent_change,
            "normality_group_p": normality_group_p,
            "test_used": test_used,
            "test_statistic": test_stat,
            "p_value": p_val
        }

    return results

