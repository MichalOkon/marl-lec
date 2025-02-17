import os
import re
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def convert_train_to_test_path(train_path):
    parts = train_path.split(os.path.sep)
    try:
        idx = parts.index("train")
        parts[idx] = "test"
        return os.path.sep.join(parts)
    except ValueError:
        return train_path.replace("train", "test")


def get_best_run_energy_logs(results_json_path):
    with open(results_json_path, 'r') as f:
        results_data = json.load(f)

    best_checkpoint_dir = results_data.get("best_checkpoint_dir")
    if best_checkpoint_dir is None:
        raise ValueError(f"'best_checkpoint_dir' not found in {results_json_path}")

    match = re.search(r'checkpoint_(\d+)', best_checkpoint_dir)
    if not match:
        raise ValueError(f"Could not extract checkpoint number from {best_checkpoint_dir}")
    checkpoint_number = match.group(1)

    train_dir = os.path.dirname(os.path.dirname(best_checkpoint_dir))

    test_dir = convert_train_to_test_path(train_dir)

    # The best run is assumed to be under test_dir/<checkpoint_number>
    run_dir = os.path.join(test_dir, checkpoint_number)
    if not os.path.isdir(run_dir):
        raise ValueError(f"Run directory {run_dir} does not exist.")

    run_results_dirs = glob.glob(os.path.join(run_dir, "run_results_*"))
    if not run_results_dirs:
        raise ValueError(f"No run_results directory found in {run_dir}")
    selected_run_dir = run_results_dirs[0]

    aggregated_csv_path = os.path.join(selected_run_dir, "aggregated_results.csv")
    if not os.path.isfile(aggregated_csv_path):
        raise ValueError(f"aggregated_results.csv not found at {aggregated_csv_path}")

    logs_df = pd.read_csv(aggregated_csv_path)
    logs_dict = logs_df.to_dict(orient='list')

    return logs_dict, aggregated_csv_path


def collect_best_run_aggregated_results(root_directory, group_prefix=None):
    results_json_files = glob.glob(os.path.join(root_directory, '**', 'results.json'), recursive=True)
    data_frames = []
    for results_json in results_json_files:
        parent_dir = os.path.basename(os.path.dirname(results_json))
        if group_prefix is not None and not parent_dir.startswith(group_prefix):
            continue  # Skip this file if it does not belong to the desired group
        try:
            _, agg_csv_path = get_best_run_energy_logs(results_json)
            df = pd.read_csv(agg_csv_path)
            data_frames.append(df)
            print(f"Loaded aggregated results from: {agg_csv_path}")
        except Exception as e:
            print(f"Error processing {results_json}: {e}")
    return data_frames


def plot_energy_logs(logs, import_prices, export_prices=None, save_path=None):
    timesteps = np.arange(len(logs['average_imported_energy']))
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(24, 12))
    bar_width = max(1.0, 1.0 * (100 / len(timesteps)))  # Adjust as needed

    # ----- Incoming Energy -----
    avg_imported_energy = np.array(logs['average_imported_energy'])
    avg_discharge = np.array(logs['average_discharge'])
    common_discharge = np.array(logs['common_battery_discharge'])
    avg_locally_imported_energy = np.array(logs['average_locally_imported_energy'])
    avg_produced_energy = np.array(logs['average_produced_energy'])

    data_first_plot = np.vstack([
        avg_imported_energy,
        avg_discharge,
        common_discharge,
        avg_locally_imported_energy,
        avg_produced_energy,
    ])

    labels_in = [
        'Average Imported Energy',
        'Average Discharge',
        'Average Common Battery Discharge',
        'Average Locally Imported Energy',
        'Average Produced Energy'
    ]
    bottom = np.zeros(len(timesteps))
    for i in range(data_first_plot.shape[0]):
        axs[0].bar(timesteps, data_first_plot[i], bottom=bottom,
                   label=labels_in[i], width=bar_width, edgecolor='none')
        bottom += data_first_plot[i]

    axs[0].set_xlabel('Timesteps')
    axs[0].set_ylabel('Energy Values [kWh]')
    axs[0].set_title('Incoming Energy')

    ax2 = axs[0].twinx()
    ax2.grid(False)
    relevant_import_prices = np.array(import_prices[:len(timesteps)])
    ax2.plot(timesteps, relevant_import_prices, color='red', linestyle='--', label='Import Prices')
    if export_prices is not None:
        relevant_export_prices = np.array(export_prices[:len(timesteps)])
        ax2.plot(timesteps, relevant_export_prices, color='green', linestyle='--', label='Export Prices')
    ax2.set_ylabel('Price [$/kWh]')
    axs[0].legend(loc='upper left')
    ax2.legend(loc='upper right')

    # ----- Outgoing Energy -----
    avg_exported_energy = np.array(logs['average_exported_energy'])
    avg_charge = np.array(logs['average_charge'])
    common_charge = np.array(logs['common_battery_charge'])
    avg_locally_exported_energy = np.array(logs['average_locally_exported_energy'])
    avg_load = np.array(logs['average_load'])

    data_second_plot = np.vstack([
        avg_exported_energy,
        avg_charge,
        common_charge,
        avg_locally_exported_energy,
        avg_load
    ])

    labels_out = [
        'Average Exported Energy',
        'Average Charge',
        'Average Common Battery Charge',
        'Average Locally Exported Energy',
        'Average Load'
    ]
    bottom = np.zeros(len(timesteps))
    for i in range(data_second_plot.shape[0]):
        axs[1].bar(timesteps, data_second_plot[i], bottom=bottom,
                   label=labels_out[i], width=bar_width, edgecolor='none')
        bottom += data_second_plot[i]

    axs[1].set_xlabel('Timesteps')
    axs[1].set_ylabel('Energy Values [kWh]')
    axs[1].set_title('Outgoing Energy')

    ax3 = axs[1].twinx()
    ax3.grid(False)
    relevant_import_prices = np.array(import_prices[:len(timesteps)])
    ax3.plot(timesteps, relevant_import_prices, color='red', linestyle='--', label='Import Prices')
    if export_prices is not None:
        relevant_export_prices = np.array(export_prices[:len(timesteps)])
        ax3.plot(timesteps, relevant_export_prices, color='green', linestyle='--', label='Export Prices')
    ax3.set_ylabel('Price [¢/kWh]')
    axs[1].legend(loc='upper left')
    ax3.legend(loc='upper right')

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=600)
    else:
        plt.show()
    plt.close(fig)


def average_and_plot_aggregated_results(root_directory, output_plot_file, import_prices_path, export_prices_path=None,
                                        group_prefix=None, num_households=1):
    data_frames = collect_best_run_aggregated_results(root_directory, group_prefix=group_prefix)
    if not data_frames:
        print("No aggregated results found for the specified group.")
        return

    numeric_dfs = []
    for df in data_frames:
        df_numeric = df.apply(pd.to_numeric, errors='coerce').select_dtypes(include=[np.number])
        numeric_dfs.append(df_numeric)

    combined_df = numeric_dfs[0].copy()
    for df in numeric_dfs[1:]:
        combined_df = combined_df.add(df, fill_value=0)
    averaged_df = combined_df / len(numeric_dfs)

    averaged_csv_path = os.path.join(root_directory, "averaged_aggregated_results_{}.csv".format(group_prefix))
    averaged_df.to_csv(averaged_csv_path, index=False)
    print(f"Averaged aggregated results saved to: {averaged_csv_path}")

    logs_dict = averaged_df.to_dict(orient='list')

    import_prices_df = pd.read_csv(import_prices_path, parse_dates=True)
    import_prices = import_prices_df['price'].values

    export_prices = None
    if export_prices_path is not None:
        export_prices_df = pd.read_csv(export_prices_path, parse_dates=True)
        export_prices = export_prices_df['price'].values

    plot_energy_logs(logs_dict, import_prices, export_prices, save_path=output_plot_file)


def run_energy_plotting(results_dir, group_prefix=None):
    root_directory = os.path.join("results", results_dir, "train")

    import_prices_path = "data/tou_price_data.csv"  # Update with your actual path.

    output_plot_file = "final_energy_logs_plot_{}.png".format(group_prefix)

    average_and_plot_aggregated_results(root_directory, output_plot_file, import_prices_path, group_prefix=group_prefix)
