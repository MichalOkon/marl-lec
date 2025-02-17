import os

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from src.utils.plotting.helper_functions import calculate_y_lim, get_ci_bootstrap


def plot_household_data(data, error_data=None, alternative_labels=None, save_path=None):
    if isinstance(data, dict):
        data = pd.DataFrame(data)

    if alternative_labels is not None:
        if len(alternative_labels) != len(data):
            raise ValueError("The length of alternative_labels must match the number of rows in data.")
        data.index = alternative_labels

    sns.set(style="whitegrid")
    fig, axs = plt.subplot_mosaic([
        [0, 0, 1, 1, 2, 2],
        ['.', 3, 3, 4, 4, '.'],
        ['.', 5, 5, 6, 6, '.']
    ], figsize=(16, 12))
    axs_list = [axs[i] for i in sorted(axs.keys())]

    top_row_columns = data.columns[:3]
    middle_row_columns = data.columns[3:5]
    bottom_row_columns = data.columns[5:]

    row_y_max = []
    row_y_min = []
    unique_labels = data.index.unique()
    palette = sns.color_palette("husl", len(unique_labels))
    color_map = dict(zip(unique_labels, palette))

    top_y_max, top_y_min = calculate_y_lim(data, top_row_columns)
    row_y_max.append(top_y_max)
    row_y_min.append(top_y_min)

    if len(middle_row_columns) > 0:
        middle_y_max, middle_y_min = calculate_y_lim(data, middle_row_columns)
        row_y_max.append(middle_y_max)
        row_y_min.append(middle_y_min)

    if len(bottom_row_columns) > 0:
        bottom_y_max, bottom_y_min = calculate_y_lim(data, bottom_row_columns)
        row_y_max.append(bottom_y_max)
        row_y_min.append(bottom_y_min)

    for i, column in enumerate(top_row_columns):
        column_data = data[[column]].dropna()
        sns.barplot(x=column_data.index, y=column_data[column], ax=axs_list[i],
                    ci=None, legend=False, hue=column_data.index,
                    palette=[color_map[label] for label in column_data.index])
        if error_data is not None:
            axs_list[i].errorbar(column_data.index, column_data[column],
                                 yerr=error_data[column], fmt='none',
                                 ecolor='black', capsize=5)
        axs_list[i].set_title(column.replace('_', ' ').title())
        axs_list[i].set_xlabel('Household Type')
        axs_list[i].tick_params(axis='x', rotation=45)
        axs_list[i].set_ylim(row_y_min[0], row_y_max[0])
        if i != 0:
            axs_list[i].set_yticklabels([])
            axs_list[i].tick_params(left=False)
            axs_list[i].set_ylabel('')
        else:
            axs_list[i].set_ylabel('Value [¢]')

    for i, column in enumerate(middle_row_columns):
        column_data = data[[column]].dropna()
        sns.barplot(x=column_data.index, y=column_data[column], ci=None, ax=axs_list[i + 3],
                    hue=column_data.index, legend=False,
                    palette=[color_map[label] for label in column_data.index])
        if error_data is not None:
            axs_list[i + 3].errorbar(column_data.index, column_data[column],
                                     yerr=error_data[column], fmt='none',
                                     ecolor='black', capsize=5)
        axs_list[i + 3].set_title(column.replace('_', ' ').title())
        axs_list[i + 3].set_xlabel('Household Type')
        axs_list[i + 3].tick_params(axis='x', rotation=45)
        axs_list[i + 3].set_ylim(0, row_y_max[1])
        if i != 0:
            axs_list[i + 3].set_yticklabels([])
            axs_list[i + 3].tick_params(left=False)
            axs_list[i + 3].set_ylabel('')
        else:
            axs_list[i + 3].set_ylabel('Value [kWh]')

    if len(bottom_row_columns) > 0:
        for i, column in enumerate(bottom_row_columns):
            column_data = data[[column]].dropna()
            sns.barplot(x=column_data.index, y=column_data[column],
                        ax=axs_list[i + 5], legend=False, hue=column_data.index,
                        palette=[color_map[label] for label in column_data.index])
            if error_data is not None:
                axs_list[i + 5].errorbar(column_data.index, column_data[column],
                                         yerr=error_data[column], fmt='none',
                                         ecolor='black', capsize=5)
            axs_list[i + 5].set_title(column.replace('_', ' ').title())
            axs_list[i + 5].set_xlabel('Household Type')
            axs_list[i + 5].tick_params(axis='x', rotation=45)
            axs_list[i + 5].set_ylim(0, row_y_max[2])
            if i != 0:
                axs_list[i + 5].set_yticklabels([])
                axs_list[i + 5].tick_params(left=False)
                axs_list[i + 5].set_ylabel('')
            else:
                axs_list[i + 5].set_ylabel('Value [kWh]')

    for j in range(len(data.columns), len(axs_list)):
        fig.delaxes(axs_list[j])
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=600)
    else:
        plt.show()
    plt.close(fig)

def read_and_plot_household_data(filepath, alternative_labels=None, save_path=None):
    data = pd.read_csv(filepath, index_col=0)
    plot_household_data(data, None, alternative_labels, save_path)


def average_accumulated_household_csvs(base_dir):
    dataframes = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("accumulated_household_logs"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                dataframes.append(df)

    if not dataframes:
        print("No accumulated_household_logs.csv files found.")
        return None

    combined_df = pd.concat(dataframes, axis=0)

    ci_df = combined_df.groupby(combined_df.columns[0], as_index=False).agg(lambda x: get_ci_bootstrap(x))
    averaged_df = combined_df.groupby(combined_df.columns[0], as_index=False).mean()
    averaged_df = averaged_df.set_index(averaged_df.columns[0])

    return averaged_df, ci_df


def plot_household_data_from_dir(base_dir, alternative_labels=None, save_path=None):
    averaged_df, error_df = average_accumulated_household_csvs(base_dir)

    if averaged_df is None:
        print("No data to plot.")
        return

    plot_household_data(averaged_df, error_df, alternative_labels, save_path)

def plot_price_data(price_data, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(price_data['time'], price_data['import_price'], label='Retail Import Price')
    ax.plot(price_data['time'], price_data['export_price'], label='Retail Export Price')

    ax.set_xlabel('Time')
    ax.set_ylabel('Price [¢/kWh]')
    ax.set_title('Retail Price Data')
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)


def read_and_plot_price_data(price_data_path, save_path=None):
    price_data = pd.read_csv(price_data_path)
    price_data['import_price'] = price_data['price']
    price_data['export_price'] = 3.0

    # Only show the first week of data
    price_data['time'] = pd.to_datetime(price_data['time'])
    price_data = price_data.loc[price_data['time'] < price_data['time'][0] + pd.Timedelta(days=7)]
    plot_price_data(price_data, save_path)
