import json
import os

import numpy as np
import pandas as pd

from src.environments.lec_resources import HouseholdResource, EnvironmentResources
from src.resources.common_battery import CommonBattery
from src.resources.gateway import Gateway
from src.resources.generator import Generator
from src.resources.load import Load
from src.resources.storage import Storage

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class PecanParser:
    def __init__(self, timeseries_file_path: str, metadata_file_path: str, prices_file_path: str = "",
                 use_cache: bool = True, pricing_mode: str = "tou", prediction_mode: str = "average",
                 battery_config_path: str = "src/config/battery_config,json", common_battery_type: str = "none"):
        self.timeseries_file_path = timeseries_file_path
        self.metadata_file_path = metadata_file_path
        self.prices_file_path = prices_file_path

        self.timeseries_data = None
        self.metadata = None
        self.price_data = None

        self.unique_household_ids = None

        # Prepare the variables
        self.household_resources = []

        self.use_cache = use_cache
        self.pricing_mode = pricing_mode

        # Can be either "average" or "past_prices"
        self.prediction_mode = prediction_mode

        self.battery_config_path = battery_config_path

        # Could be default, realistic or none
        self.common_battery_type = common_battery_type
        return

    def parse(self):
        # Check if the specs are cached
        if self.use_cache and self.is_cache_present():
            self.load_processed_data()
            return

        # Else parse the files
        self.read_files()
        self.process_tables()
        self.save_processed_data()
        # For testing
        self.print_tables()
        return

    def get_parsed_resources(self, household_ids=None, start_time=None, end_time=None, num_timesteps=None):
        if start_time is None:
            start_time = self.timeseries_data['time'].iloc[0]
        if end_time is None:
            end_time = self.timeseries_data['time'].iloc[-1]

        relevant_price_data = self.price_data[
            (self.price_data['time'] >= start_time) & (self.price_data['time'] <= end_time)]

        if num_timesteps is not None:
            relevant_price_data = relevant_price_data.head(num_timesteps)
            end_time = relevant_price_data['time'].iloc[-1]
            end_time = pd.to_datetime(end_time) + pd.Timedelta(15, 'm')
            end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')

        relevant_timeseries_data = self.timeseries_data[
            (self.timeseries_data['time'] >= start_time) & (self.timeseries_data['time'] <= end_time)]

        resources = self.create_resources(ids=household_ids, price_data=relevant_price_data,
                                          timeseries_data=relevant_timeseries_data)
        return resources

    def read_files(self):
        self.timeseries_data = pd.read_csv(self.timeseries_file_path)
        self.metadata = pd.read_csv(self.metadata_file_path)
        # Interpret ; as the delimiter and , as the decimal point
        # The prices are given as cents per kWh
        if self.pricing_mode == "synthetic":
            self.price_data = pd.read_csv('data/synthetic_price_data.csv')
        elif self.pricing_mode == "tou":
            self.price_data = pd.read_csv('data/tou_price_data.csv')
        else:
            self.price_data = pd.read_csv(self.prices_file_path, delimiter=';', decimal=',')
        return

    def process_real_price_data(self):
        self.price_data = self.price_data[['DateTime', 'realtime_lbmp (avg) (nyiso)']]
        self.price_data.rename(columns={'DateTime': 'time', 'realtime_lbmp (avg) (nyiso)': 'price'}, inplace=True)

        # Resample the price data to 15 minutes
        self.price_data['time'] = pd.to_datetime(self.price_data['time'])
        self.price_data.set_index('time', inplace=True)
        self.price_data = self.price_data.resample('15T').mean().reset_index()

        self.price_data['time'] = pd.to_datetime(self.price_data['time'])

    def process_price_data(self):
        if self.pricing_mode != "synthetic" and self.pricing_mode != "tou":
            self.process_real_price_data()
        return

    def process_metadata(self):
        self.metadata = self.metadata[self.metadata['dataid'].isin(self.unique_household_ids)]
        self.metadata = self.metadata[['dataid', 'pv']]

        self.metadata.rename(columns={'dataid': 'id'}, inplace=True)
        self.metadata.reset_index(drop=True, inplace=True)

    def process_timeseries_data(self):
        self.timeseries_data = self.timeseries_data[['dataid', 'local_15min', 'grid', 'solar', 'solar2']]

        self.timeseries_data.rename(columns={'dataid': 'id', 'local_15min': 'time'}, inplace=True)

        self.timeseries_data['time'] = pd.to_datetime(self.timeseries_data['time'])

        print(self.timeseries_data.head())
        self.timeseries_data.sort_values(by=['id', 'time'], inplace=True)

        filled_solar1 = self.timeseries_data['solar'].fillna(0)
        filled_solar2 = self.timeseries_data['solar2'].fillna(0)
        self.timeseries_data['usage'] = self.timeseries_data['grid'] + filled_solar1 + filled_solar2
        self.timeseries_data['usage'] = np.maximum(self.timeseries_data['usage'], 0)
        self.timeseries_data['total_solar'] = filled_solar1 + filled_solar2

        self.timeseries_data['time'] = pd.to_datetime(self.timeseries_data['time'])
        self.timeseries_data.reset_index(drop=True, inplace=True)

        resampled_data = []
        unique_household_ids = self.timeseries_data['id'].unique()
        for id in unique_household_ids:
            id_mask = self.timeseries_data['id'] == id
            household_data = self.timeseries_data[id_mask]

            household_data['time'] = pd.to_datetime(household_data['time'])
            household_data = household_data.set_index('time').resample('15T').asfreq().reset_index()
            household_data['id'] = id

            household_data = household_data.fillna(0)

            resampled_data.append(household_data)

        self.timeseries_data = pd.concat(resampled_data).reset_index(drop=True)

    def process_tables(self):

        self.unique_household_ids = self.timeseries_data['dataid'].unique()

        self.process_price_data()
        self.process_timeseries_data()
        self.process_metadata()
        self.add_predictions()

        return

    def add_predictions(self):
        # Add predictions of prices and pv generation for each timestep as columns
        # Predictions are generated as the average of the same hour of the previous 7 days
        self.price_data['price_prediction'] = np.nan
        self.timeseries_data['pv_prediction'] = np.nan
        prediction_start = 4 * 24 * 7
        interval = 4 * 24
        for id in self.unique_household_ids:
            id_mask = self.timeseries_data['id'] == id
            household_row_count = np.sum(id_mask)
            # Continue if the household does not have pv
            if (self.metadata[self.metadata['id'] == id]['pv'] != 'yes').any() or pd.isna(
                    self.timeseries_data.loc[id_mask, 'total_solar'].iloc[0]):
                continue
            # Start iterating after 7 days
            for i in range(prediction_start, household_row_count):
                # print(f"Predicting solar for timestep {i} and household {id}")
                mean_past_pv = self.timeseries_data.loc[id_mask, 'total_solar'].iloc[
                               i - prediction_start:i:interval].mean()
                actual_index = self.timeseries_data[id_mask].index[i]
                self.timeseries_data.at[actual_index, 'pv_prediction'] = mean_past_pv

        for i in range(prediction_start, len(self.price_data)):
            # Get dataframes with the same id in the previous 7 days
            # print("Predicting price for timestep", i)

            mean_past_prices = self.price_data.iloc[i - prediction_start:i:interval]['price'].mean()
            self.price_data.at[i, 'price_prediction'] = mean_past_prices

        return

    def create_gateway(self, price_data, timeseries_data):
        minimal_sell_price = 3.0
        sell_prices = np.minimum(minimal_sell_price, price_data['price']).reset_index(drop=True)
        gateway = Gateway(
            timestamps=timeseries_data['time'],
            imports=np.zeros(price_data['price'].shape),
            exports=np.zeros(price_data['price'].shape),
            import_price=price_data['price'].values,
            export_price=sell_prices.values,
            imported_cost=np.zeros(price_data['price'].shape),
            exported_cost=np.zeros(price_data['price'].shape),
            price_prediction=price_data['price_prediction'].values
        )
        return gateway

    def create_load(self, household_ts):
        load_values = household_ts['usage'].values
        return Load(value=load_values)

    def create_generator(self, household_id, household_ts, metadata):
        has_pv = (metadata[metadata['id'] == household_id]['pv'] == 'yes').any() and \
                 not pd.isna(household_ts['solar'].iloc[0])
        if has_pv:
            pv_values = household_ts['total_solar'].values
            pv_prediction = household_ts['pv_prediction'].values
            return Generator(
                production=np.zeros(pv_values.shape),
                upper_bound=pv_values,
                generation_prediction=pv_prediction
            )
        return None

    def load_battery_config(self):
        print("Current working directory:", os.getcwd())
        print("Trying to access:", os.path.abspath(self.battery_config_path))
        if os.path.exists(os.path.abspath(self.battery_config_path)):
            with open(os.path.abspath(self.battery_config_path), "r") as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Config file {os.path.abspath(self.battery_config_path)} not found.")

    def create_storage(self, household_id, household_ts):
        battery_config_data = self.load_battery_config()
        household_str = str(household_id)
        battery_profile = {'present': False}

        if "household_mapping" in battery_config_data and household_str in battery_config_data["household_mapping"]:
            battery_type = battery_config_data["household_mapping"][household_str]
            battery_profile = battery_config_data["battery_types"].get(battery_type, {'present': False})

        if battery_profile.get('present', False):
            shape = household_ts['usage'].values.shape
            return Storage(
                soc=np.zeros(shape),
                capacity_max=battery_profile['capacity_max'],
                initial_charge=battery_profile['initial_charge'],
                discharge_efficiency=battery_profile['discharge_efficiency'],
                charge_efficiency=battery_profile['charge_efficiency'],
                discharge_max=battery_profile['p_discharge_max'],
                charge_max=battery_profile['p_charge_max'],
                charge=np.zeros(shape),
                discharge=np.zeros(shape)
            )

        return None

    def create_common_battery(self, shape):
        if self.common_battery_type == "none":
            return None

        battery_config_data = self.load_battery_config()
        common_batteries = battery_config_data.get("common_battery_types", {})
        common_battery_config = common_batteries.get(self.common_battery_type, {})
        if not common_battery_config:
            raise ValueError(f"Common battery type '{self.common_battery_type}' not found in configuration.")
        return CommonBattery(
            soc=np.zeros(shape),
            capacity=common_battery_config.get("capacity", 100),
            charge_efficiency=common_battery_config.get("charge_efficiency", 1.0),
            max_charge_power=common_battery_config.get("max_charge_power", 1.25),
            max_discharge_power=common_battery_config.get("max_discharge_power", 1.25),
            charge_vals=np.zeros(shape),
            discharge_vals=np.zeros(shape)
        )

    def create_resources(self, ids=None, timeseries_data=None, metadata=None, price_data=None):
        if timeseries_data is None:
            timeseries_data = self.timeseries_data
        if metadata is None:
            metadata = self.metadata
        if price_data is None:
            price_data = self.price_data

        shared_gateway = self.create_gateway(price_data, timeseries_data)

        common_battery = self.create_common_battery(price_data['price'].shape)

        env_resources = EnvironmentResources(gateway=shared_gateway, common_battery=common_battery)

        relevant_ids = ids if ids is not None else self.unique_household_ids

        for i, household_id in enumerate(relevant_ids):
            household_ts = timeseries_data[timeseries_data['id'] == household_id].copy()
            household_ts['time'] = pd.to_datetime(household_ts['time'])
            household_ts.set_index('time', inplace=True)

            expected = household_ts.resample('15min').asfreq()
            missing_timestamps = expected[expected['usage'].isnull()]
            print(f"Missing timestamps for household {household_id}:")
            print(missing_timestamps)
            print(f"Length of household {household_id} timeseries: {len(household_ts)}")

            load_resource = self.create_load(household_ts)
            generator_resource = self.create_generator(household_id, household_ts, metadata)
            storage_resource = self.create_storage(household_id, household_ts)

            household_resource = HouseholdResource(
                household_id=household_id,
                load=load_resource,
                generator=generator_resource,
                storage=storage_resource
            )

            env_resources.add_household(household_resource, i)

        print(env_resources.households.keys())
        return env_resources

    def check_if_pricing_data_processed(self):
        if self.pricing_mode == "synthetic":
            return self.check_if_file_exists('data/synthetic_price_data.csv')
        elif self.pricing_mode == "tou":
            return self.check_if_file_exists('data/tou_price_data.csv')
        else:
            return self.check_if_file_exists('data/pecan_street/cache/processed_price_data.csv')

    def is_cache_present(self):
        return self.check_if_file_exists(
            'data/pecan_street/cache/processed_timeseries_data.csv') and self.check_if_file_exists(
            'data/pecan_street/cache/processed_price_data.csv') and self.check_if_pricing_data_processed()

    def save_processed_data(self):
        # Save processed tables to csv files
        self.timeseries_data.to_csv('data/pecan_street/cache/processed_timeseries_data.csv', index=False)
        self.metadata.to_csv('data/pecan_street/cache/processed_metadata.csv', index=False)
        self.price_data.to_csv('data/pecan_street/cache/processed_price_data.csv', index=False)

        return

    def load_processed_data(self):
        # Load processed tables from csv files
        self.timeseries_data = pd.read_csv('data/pecan_street/cache/processed_timeseries_data.csv')
        self.metadata = pd.read_csv('data/pecan_street/cache/processed_metadata.csv')
        if self.pricing_mode == "synthetic":
            self.price_data = pd.read_csv('data/synthetic_price_data.csv')
        elif self.pricing_mode == "tou":
            self.price_data = pd.read_csv('data/tou_price_data.csv')
        else:
            self.price_data = pd.read_csv('data/pecan_street/cache/processed_price_data.csv')
        self.unique_household_ids = self.timeseries_data['id'].unique()
        return

    @staticmethod
    def check_if_file_exists(filename):
        try:
            with open(filename,
                      'r') as f:
                return True
        except FileNotFoundError:
            return False

    @staticmethod
    def get_start_end_times_from_scenario(scenario):
        scenario_times_mapping = {
            'scarce': ('2019-05-28', '2019-06-01'),
            'abundant': ('2019-05-25', '2019-05-29'),
            'testing': ('2019-05-01', '2019-05-16'),  # The one used for testing
            'short_testing': ('2019-05-11', '2019-05-15'),
            'long_testing': ('2019-06-10', '2019-06-18'),
            'matching': ('2019-05-21', '2019-05-25'),
            'long': ('2019-06-01', '2019-06-16'),
            'medium': ('2019-06-01', '2019-06-09'),
            'final': ('2019-05-02', '2019-05-10'),
            'final_short': ('2019-05-04', '2019-05-08'),
            'month': ('2019-10-01', '2019-11-01'),
            'month_testing': ('2019-05-01', '2019-06-02'),
            'few_months': ('2019-05-16', '2019-11-01')  # The one actually used in training
        }
        return scenario_times_mapping.get(scenario, (None, None))

    def get_preset_resources(self, preset, used_household_ids):
        start_time, end_time = self.get_start_end_times_from_scenario(preset)
        return self.get_parsed_resources(used_household_ids, start_time, end_time)

    def print_tables(self):
        # Print first few rows of the tables
        print("Timeseries data:")
        print(self.timeseries_data.head(6))
        print("Metadata:")
        print(self.metadata.head())
        print("Price data:")
        print(self.price_data.head())

    def plot_tables(self, start_time=None, end_time=None):
        self.price_data['time'] = pd.to_datetime(self.price_data['time']).dt.tz_localize(None)
        self.timeseries_data['time'] = pd.to_datetime(self.timeseries_data['time']).dt.tz_localize(None)

        if start_time is None:
            start_time = self.timeseries_data['time'].iloc[0]
        if end_time is None:
            end_time = self.timeseries_data['time'].iloc[-1]

        price_data = self.price_data[(self.price_data['time'] >= start_time) & (self.price_data['time'] <= end_time)]
        timeseries_data = self.timeseries_data[
            (self.timeseries_data['time'] >= start_time) & (self.timeseries_data['time'] <= end_time)]

        fig, ax = plt.subplots(figsize=(10, 15))
        ax.plot(price_data['time'], price_data['price'], label='Price')
        ax.legend()

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        fig.autofmt_xdate()
        plt.show()

        for id in self.unique_household_ids:
            if (self.metadata[self.metadata['id'] == id]['pv'] == 'yes').any():
                household_timeseries = timeseries_data[timeseries_data['id'] == id]
                fig, ax = plt.subplots(figsize=(10, 15))

                print(f"Plotting solar data for household {id}")
                ax.plot(household_timeseries['time'], household_timeseries['pv_prediction'],
                        label=f'Household {id} pv prediction')
                ax.plot(household_timeseries['time'], household_timeseries['total_solar'],
                        label=f'Household {id} solar', alpha=0.5)
                ax.legend()

                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

                fig.autofmt_xdate()
                plt.show()
            else:
                print(f"No solar data for household {id}")

    def find_extreme_price_periods(self, smallest=False, n=3):
        data_copy = self.price_data.copy()
        data_copy = data_copy.dropna(subset=['price_prediction'])

        if 'time' not in data_copy.index.names:
            data_copy.set_index('time', inplace=True)

        data_copy['n_day_avg'] = data_copy['price'].rolling(288).mean()

        if smallest:
            found_periods = data_copy['n_day_avg'].nsmallest(n)
        else:
            found_periods = data_copy['n_day_avg'].nlargest(n)

        found_periods_data = data_copy.loc[found_periods.index]
        found_periods_data.reset_index(inplace=True)
        return found_periods_data

    def get_average_price(self):
        return self.price_data['price'].mean()

    def calculate_baseline_costs_with_solar(self, household_ids=None, pv_efficiency=1.0, start_time=None, end_time=None,
                                            scenario=None, num_timesteps=None):
        if scenario is not None:
            start_time, end_time = self.get_start_end_times_from_scenario(scenario)
        else:
            if start_time is None:
                start_time = self.timeseries_data['time'].iloc[0]
            if end_time is None:
                end_time = self.timeseries_data['time'].iloc[-1]
        if household_ids is None:
            household_ids = self.unique_household_ids

        resources = self.get_parsed_resources(household_ids, start_time=start_time, end_time=end_time,
                                              num_timesteps=num_timesteps)
        total_cost_per_household = {}

        all_households_cost = 0
        for household_id in resources.households.keys():
            household_resources = resources.households[household_id]
            load = household_resources.load
            if household_resources.generator is not None:
                generation = household_resources.generator.upper_bound * pv_efficiency
                generation_surplus = np.maximum(generation - load.value, 0)
            else:
                generation = np.zeros(load.value.shape)
                generation_surplus = np.zeros(load.value.shape)
            import_price = resources.gateway.import_price
            # Do not include export price if it is the training environment since we do not count it towards the cost
            export_price = resources.gateway.export_price if "testing" in scenario else 0
            generation_deficit = np.maximum(load.value - generation, 0)
            total_cost = np.sum(generation_deficit * import_price) - np.sum(generation_surplus * export_price)
            total_cost_per_household[household_id] = total_cost
            all_households_cost += total_cost

        return total_cost_per_household, all_households_cost / len(resources.households.keys())
