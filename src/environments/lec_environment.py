import json
import random
import time
import os
from collections import defaultdict
from threading import Lock

import gymnasium as gym
import numpy as np
from copy import deepcopy

import pandas as pd

from src.environments.lec_resources import EnvironmentResources, HouseholdResource
from src.resources import Gateway
from src.utils.plotting.households_plots import read_and_plot_household_data
from src.utils.plotting.energy_plots import plot_energy_logs
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import logging

# Configure logging
formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
logging.basicConfig(filename='logs/training_info_logs_{}.log'.format(formatted_time), level=logging.INFO, filemode='w',
                    format='%(asctime)s - %(message)s')


class LECEnvironment(MultiAgentEnv):
    metadata = {'name': 'EnergyCommunityMultiHouseholdsEnv_v1'}

    def __init__(self,
                 env_resources: EnvironmentResources,
                 saving_dir,
                 pv_efficiency=1.0,
                 max_timesteps=None,
                 trading_phases=1,
                 forecasting=True,
                 printing=True,
                 log_debugging=True,
                 reward_export=False,
                 common_reward_factor=0.0,
                 price_based_common_reward_factor=False,
                 auto_plotting=False):
        super().__init__()

        self.saving_dir = saving_dir
        self.printing = printing
        self.log_debugging = log_debugging

        self.auto_plotting = auto_plotting

        # Store the environment resources
        self.env_resources = deepcopy(env_resources)
        self.households = self.env_resources.households
        self.common_battery = self.env_resources.common_battery
        self.gateway = self.env_resources.gateway

        self.num_trading_phases = trading_phases
        self.forecasting_range = 16
        self.reward_export = reward_export

        # Determine maximum timesteps from one of the household load series
        if max_timesteps is None:
            self.max_timesteps = self._calculate_max_timestep()
        else:
            self.max_timesteps = max_timesteps

        self.forecasting_step = 4 if forecasting else 0

        if isinstance(pv_efficiency, (float, int)):
            pv_efficiency = [pv_efficiency] * len(self.households)
        self.pv_efficiency = pv_efficiency

        self.households = self._create_households()
        self.households_ids = set(self.households)
        self.num_households = len(self.households)

        self._create_action_space()
        self._create_observation_space()

        self.current_timestep: int = 0
        self.current_total_production: float = 0
        self.current_total_consumption: float = 0
        self.local_market_price = 0

        self.current_offers_quantities = {}
        self.historic_offers_prices = []
        self.historic_offers_quantities = []
        self.historic_traded_logs = defaultdict(list)
        self.balance_history = []
        self.reward_history = []

        self.common_reward_factor = common_reward_factor
        self.price_based_common_reward_factor = price_based_common_reward_factor
        self.current_real_reward = {}
        self.rewards = defaultdict(float)

        print("Common reward factor: ", self.common_reward_factor)

    def _create_households(self):
        households = {}
        for i, (household_id, household_resource) in enumerate(self.env_resources.households.items()):
            households[household_id] = Household(
                id=household_id,
                resources=household_resource,
                gateway=self.gateway,
                num_household=len(self.households),
                pv_efficiency=self.pv_efficiency[i],
                num_trading_phases=self.num_trading_phases,
                forecasting_step=self.forecasting_step,
                reward_export=self.reward_export,
                has_common_battery=(self.common_battery is not None),
                printing_debug=self.printing,
                log_debugging=self.log_debugging
            )
        return households

    def _create_action_space(self):
        self._action_space_in_preferred_format = True
        temp_action_space = {}
        for id in self.households:
            temp_action_space[id] = self.households[id].action_space

        self.action_space = gym.spaces.Dict(temp_action_space)

    def _create_observation_space(self):
        self._observation_space_in_preferred_format = True
        temp_observation_space = {}
        for id in self.households:
            temp_observation_space[id] = self.households[id].observation_space

        self.observation_space = gym.spaces.Dict(temp_observation_space)

    def reset(self, *, seed=None, options=None):
        self.env_resources = deepcopy(self.env_resources)
        self.households = self.env_resources.households
        self.common_battery = self.env_resources.common_battery
        self.gateway = self.env_resources.gateway

        self.households = self._create_households()
        self.households_ids = set(self.households)

        self.current_timestep = 0
        self.current_total_production = 0
        self.current_total_consumption = 0
        self.local_market_price = 0
        self.balance_history = []
        self.reward_history = []
        self.historic_offers_prices = []
        self.historic_offers_quantities = []
        self.historic_traded_logs = defaultdict(list)
        self.current_offers_quantities = {}
        self.current_real_reward = {}

        observations = self._get_households_observations()
        return observations, {}

    def step(self, action_dict: dict) -> tuple:
        exists_actions = len(action_dict) > 0

        observations = {}
        self.rewards: dict = defaultdict(float)
        info: dict = {}
        if self.current_timestep >= self.max_timesteps:
            observations, reward = {}, self._give_final_reward()
            self._add_rewards(reward)
            termination_flags, truncation_flags = self._set_termination_flags(True)
            self._log_and_store_episode_end_values()

        elif exists_actions:
            self._set_energy_demand(action_dict)

            household = next(iter(self.households.values()))
            self.local_market_price = (household.gateway.export_price[self.current_timestep] +
                                       household.gateway.import_price[self.current_timestep]) / 2

            if self.num_trading_phases > 0:
                reward, info = self._manage_market_actions(action_dict)
                self._add_rewards(reward)

            if self.common_battery is not None:
                common_battery_rewards = self._manage_common_battery(action_dict)
                self._add_rewards(common_battery_rewards)

            if self.common_battery is not None:
                self.common_battery.soc[self.current_timestep] = self.common_battery.soc[self.current_timestep]

            step_rewards = {}
            for household_id in action_dict:
                step_rewards[household_id], info[household_id] = self.households[household_id].step(
                    action_dict[household_id])
                self.print_info("Reward for household {}: {}".format(household_id, step_rewards[household_id]))

                assert not np.isnan(step_rewards[household_id]), "Reward is NaN for household {}".format(household_id)
            self._add_rewards(step_rewards)

            termination_flags, truncation_flags = self._set_termination_flags(False)

            self._iterate_timestep()
            if self.current_timestep >= self.max_timesteps:
                info = {}
            observations = self._get_households_observations()
        else:
            termination_flags, truncation_flags = self._set_termination_flags(False)

        return observations, self.rewards, termination_flags, truncation_flags, info

    def _add_rewards(self, new_rewards: dict):
        for household_id in new_rewards:
            self.rewards[household_id] += new_rewards[household_id]

    def _give_final_reward(self):
        rewards = {}
        for household_id in self.households.keys():
            avg_energy_price = np.mean(self.households[household_id].gateway.export_price)
            self.households[household_id].current_available_energy = 0
            if self.households[household_id].storage is not None:
                rewards[household_id] = self.households[household_id].storage.soc[-1] * avg_energy_price
            else:
                rewards[household_id] = 0
        return rewards

    def _set_energy_demand(self, action_dict: dict):
        for household_id in action_dict:
            self.households[household_id].current_available_energy = -self.households[household_id].load.value[
                self.current_timestep]
            self.households[household_id].current_available_energy += self.households[household_id].exchanged_energy
            self.households[household_id].exchanged_energy = 0

    def _manage_common_battery(self, action_dict: dict):
        if self.common_battery is None:
            return

        # Randomize the order of execution to avoid bias
        household_ids = list(action_dict.keys())
        random.shuffle(household_ids)

        rewards = defaultdict(float)
        for household_id in household_ids:
            current_energy_price = self.households[household_id].gateway.import_price[self.current_timestep]

            if action_dict[household_id]['common_battery_action_type'] == 0:
                continue
            elif action_dict[household_id]['common_battery_action_type'] == 1:
                charge = action_dict[household_id]['common_battery_action_value'][0]
                final_charge = charge * self.common_battery.max_charge_power / self.common_battery.capacity

                if self.common_battery.soc[self.current_timestep] + final_charge > 1.0:
                    final_charge = 1.0 - self.common_battery.soc[self.current_timestep]

                used_energy = final_charge * self.common_battery.capacity / self.common_battery.charge_efficiency

                self.households[household_id].current_available_energy -= used_energy
                self.common_battery.soc[self.current_timestep] += final_charge
                energy_charged = final_charge * self.common_battery.capacity
                self.common_battery.charge_vals[self.current_timestep] += energy_charged
                self.households[household_id].charge_common_battery[self.current_timestep] += energy_charged
                self.print_info(
                    f"Household {household_id} charged the common battery with {energy_charged} units of energy")

                if self.price_based_common_reward_factor:
                    rewards[household_id] += energy_charged * current_energy_price * self.common_reward_factor
                else:
                    rewards[household_id] += energy_charged * self.common_reward_factor

            elif action_dict[household_id]['common_battery_action_type'] == 2:
                discharge = action_dict[household_id]['common_battery_action_value'][0]
                final_discharge = discharge * self.common_battery.max_discharge_power / self.common_battery.capacity

                if self.common_battery.soc[self.current_timestep] - final_discharge < 0.0:
                    # Calculate the deviation from the bounds
                    final_discharge = self.common_battery.soc[self.current_timestep]
                retrieved_energy = final_discharge * self.common_battery.capacity

                self.households[household_id].current_available_energy += retrieved_energy
                self.common_battery.soc[self.current_timestep] -= final_discharge

                self.common_battery.discharge_vals[self.current_timestep] += retrieved_energy
                self.households[household_id].discharge_common_battery[self.current_timestep] += retrieved_energy
                self.print_info(
                    f"Household {household_id} discharged the common battery with {retrieved_energy} units of energy")

                if self.price_based_common_reward_factor:
                    rewards[household_id] -= retrieved_energy * current_energy_price * self.common_reward_factor
                else:
                    rewards[household_id] -= retrieved_energy * self.common_reward_factor

        return rewards

    def _manage_market_actions(self, actions: dict) -> tuple[dict, dict]:
        self._enter_offers(actions)
        self._append_market_info()
        cost_per_household = self._exchange_energy()
        market_rewards_per_household = self._calculate_market_rewards(cost_per_household)

        info = {}
        return market_rewards_per_household, info

    def _enter_offers(self, actions: dict) -> None:
        for household_id in actions:
            # 0 -> no trade, 1 -> buy, 2 -> sell
            if actions[household_id]['trade_action'] == 0:
                self.current_offers_quantities[household_id] = 0
            elif actions[household_id]['trade_action'] == 1:
                self.current_offers_quantities[household_id] = -actions[household_id]['current_offers_quantity'][0]
            elif actions[household_id]['trade_action'] == 2:
                self.current_offers_quantities[household_id] = actions[household_id]['current_offers_quantity'][0]

    def _exchange_energy(self) -> dict:
        household_ids = list(range(self.num_households))

        # Split the offers into bids (buyers) and asks (sellers)
        bids = [(abs(quantity), household_id) for household_id, quantity in self.current_offers_quantities.items() if
                quantity < 0]
        asks = [(quantity, household_id) for household_id, quantity in self.current_offers_quantities.items() if
                quantity > 0]

        total_demand = sum([quantity for quantity, _ in bids])
        total_supply = sum([quantity for quantity, _ in asks])

        cost_per_household = defaultdict(float)
        # Fixed energy price for the market is the average of the import and export costs (mid-market price)
        fixed_price = self.local_market_price
        allocation_to_buyers = {}
        allocation_from_sellers = {}

        # Case 0: No trading
        if total_demand == 0 or total_supply == 0:
            return cost_per_household

        # Case 1: Total Supply equals Total Demand
        if total_supply == total_demand:
            for quantity, household_id in bids:
                allocation_to_buyers[household_id] = quantity
            for quantity, household_id in asks:
                allocation_from_sellers[household_id] = quantity

        # Case 2: Total Supply exceeds Total Demand
        elif total_supply > total_demand:
            for quantity, household_id in bids:
                allocation_to_buyers[household_id] = quantity
            for quantity, household_id in asks:
                proportion = quantity / total_supply
                allocated_quantity = proportion * total_demand
                allocation_from_sellers[household_id] = allocated_quantity

        # Case 3: Total Demand exceeds Total Supply
        else:
            for quantity, household_id in asks:
                allocation_from_sellers[household_id] = quantity
            for quantity, household_id in bids:
                proportion = quantity / total_demand
                allocated_quantity = proportion * total_supply
                allocation_to_buyers[household_id] = allocated_quantity

        # Update households' energy and costs
        # Process sellers
        for household_id, allocated_quantity in allocation_from_sellers.items():
            self.households[household_id].exchanged_energy -= allocated_quantity
            self.households[household_id].local_exports[self.current_timestep] += allocated_quantity

            transaction_cost = allocated_quantity * fixed_price
            cost_per_household[household_id] -= transaction_cost
            self.households[household_id].accumulated_market_cost -= transaction_cost
            self.households[household_id].balance[self.current_timestep] += transaction_cost

            self.historic_traded_logs[self.current_timestep].append(
                (float(fixed_price), float(allocated_quantity), household_id, 'Buyer(s)'))

        # Process buyers
        for household_id, allocated_quantity in allocation_to_buyers.items():
            self.households[household_id].exchanged_energy += allocated_quantity
            self.households[household_id].local_imports[self.current_timestep] += allocated_quantity

            transaction_cost = allocated_quantity * fixed_price
            cost_per_household[household_id] += transaction_cost
            self.households[household_id].accumulated_market_cost += transaction_cost
            self.households[household_id].balance[self.current_timestep] -= transaction_cost

            self.historic_traded_logs[self.current_timestep].append(
                (float(fixed_price), float(allocated_quantity), 'Seller(s)', household_id))

        # Allocations logging
        for household_id in household_ids:
            if household_id in allocation_to_buyers:
                allocated_qty = allocation_to_buyers[household_id]
                self.print_info(
                    f"Buyer Household {household_id} allocated {allocated_qty} units of energy at price {fixed_price}")
            if household_id in allocation_from_sellers:
                allocated_qty = allocation_from_sellers[household_id]
                self.print_info(
                    f"Seller Household {household_id} allocated {allocated_qty} units of energy at price {fixed_price}")

        return cost_per_household

    def _calculate_market_rewards(self, cost_per_household) -> dict:
        rewards = {}
        for household_id in self.households:
            rewards[household_id] = - cost_per_household[household_id]
            if np.isnan(rewards[household_id]):
                logging.error(
                    f"Calculated reward is NaN")
        return rewards

    def _iterate_timestep(self):
        self.current_timestep += 1
        for household_id in self.households.keys():
            self.households[household_id].current_timestep = self.current_timestep

    def _get_households_observations(self) -> dict:
        observations = {}

        if self.current_timestep >= self.max_timesteps:
            return observations

        for household_id in self.households.keys():
            if self.current_timestep == 0:
                observations[household_id] = self.households[household_id].get_initial_observations()
            else:
                observations[household_id] = self.households[household_id].get_next_observations()
                if self.common_battery is not None:
                    observations[household_id]['current_common_soc'] = np.array(
                        [self.common_battery.soc[self.current_timestep]],
                        dtype=np.float32)
        return observations

    # Log ending of episode
    def _set_termination_flags(self, is_over_flag: bool) -> tuple[dict, dict]:
        terminateds = {a: is_over_flag for a in self.households}
        terminateds['__all__'] = is_over_flag
        truncateds = {a: is_over_flag for a in self.households}
        truncateds['__all__'] = is_over_flag

        if is_over_flag:
            self._log_and_store_episode_end_values()

        return terminateds, truncateds

    @staticmethod
    def get_current_time():
        return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    def create_results_directory(self, current_time):

        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)

        # Find the subdirectory with the highest number
        subdirs = [int(name) for name in os.listdir(self.saving_dir) if
                   os.path.isdir(os.path.join(self.saving_dir, name)) and name.isdigit()]
        if len(subdirs) > 0:
            iteration = max(subdirs)
        else:
            iteration = 0

        dir_name = self.saving_dir + "/{}".format(iteration) + "/run_results_{}".format(current_time)
        os.makedirs(dir_name, exist_ok=True)
        return dir_name

    def calculate_aggregated_values(self):
        num_households = len(self.households)
        max_timesteps = self.max_timesteps

        # Initialize arrays to collect per-household data
        imports_array = np.zeros((num_households, max_timesteps))
        exports_array = np.zeros((num_households, max_timesteps))
        local_imports_array = np.zeros((num_households, max_timesteps))
        local_exports_array = np.zeros((num_households, max_timesteps))
        load_array = np.zeros((num_households, max_timesteps))
        generator_array = np.zeros((num_households, max_timesteps))
        storage_value_array = np.zeros((num_households, max_timesteps))
        storage_charge_array = np.zeros((num_households, max_timesteps))
        storage_discharge_array = np.zeros((num_households, max_timesteps))

        # Loop over households to collect data
        for idx, household_id in enumerate(self.households):
            household = self.households[household_id]
            imports_array[idx] = household.gateway.imports[:max_timesteps]
            exports_array[idx] = household.gateway.exports[:max_timesteps]
            local_imports_array[idx] = household.local_imports[:max_timesteps]
            local_exports_array[idx] = household.local_exports[:max_timesteps]
            load_array[idx] = household.load.value[:max_timesteps]

            if household.generator is not None:
                generator_array[idx] = household.generator.production[:max_timesteps]

            if household.storage is not None:
                storage_value_array[idx] = household.storage.soc[:max_timesteps]
                storage_charge_array[idx] = household.storage.charge[:max_timesteps]
                storage_discharge_array[idx] = household.storage.discharge[:max_timesteps]

        # Sum over households
        total_imported_energy = np.sum(imports_array, axis=0)
        total_exported_energy = np.sum(exports_array, axis=0)
        total_locally_imported_energy = np.sum(local_imports_array, axis=0)
        total_locally_exported_energy = np.sum(local_exports_array, axis=0)
        total_load = np.sum(load_array, axis=0)
        total_produced_energy = np.sum(generator_array, axis=0)
        total_soc = np.sum(storage_value_array, axis=0)
        total_charge = np.sum(storage_charge_array, axis=0)
        total_discharge = np.sum(storage_discharge_array, axis=0)

        # Compute averages
        average_imported_energy = total_imported_energy / num_households
        average_exported_energy = total_exported_energy / num_households
        average_locally_imported_energy = total_locally_imported_energy / num_households
        average_locally_exported_energy = total_locally_exported_energy / num_households
        average_load = total_load / num_households
        average_produced_energy = total_produced_energy / num_households
        average_soc = total_soc / num_households
        average_charge = total_charge / num_households
        average_discharge = total_discharge / num_households

        logs = {
            'average_imported_energy': average_imported_energy.tolist(),
            'average_exported_energy': average_exported_energy.tolist(),
            'average_soc': average_soc.tolist(),
            'average_charge': average_charge.tolist(),
            'average_discharge': average_discharge.tolist(),
            'average_produced_energy': average_produced_energy.tolist(),
            'average_locally_imported_energy': average_locally_imported_energy.tolist(),
            'average_locally_exported_energy': average_locally_exported_energy.tolist(),
            'average_load': average_load.tolist(),
        }

        return logs

    def calculate_household_values(self):
        accumulated_household_logs = {}
        for household_id in self.households:
            accumulated_household_logs[household_id] = {}
            accumulated_retail_import_cost = np.sum(self.households[household_id].gateway.imported_cost)
            accumulated_retail_export_profit = -np.sum(self.households[household_id].gateway.exported_cost)

            accumulated_household_logs[household_id]['accumulated_import_cost'] = accumulated_retail_import_cost
            accumulated_household_logs[household_id]['accumulated_export_profit'] = accumulated_retail_export_profit
            accumulated_household_logs[household_id]['accumulated_market_balance'] = self.households[
                household_id].accumulated_market_cost

            if self.common_battery is not None:
                accumulated_household_logs[household_id]['total_common_battery_discharge'] = np.sum(
                    self.common_battery.discharge_vals)
                accumulated_household_logs[household_id]['total_common_battery_charge'] = np.sum(
                    self.common_battery.charge_vals)
            else:
                accumulated_household_logs[household_id]['total_common_battery_discharge'] = 0
                accumulated_household_logs[household_id]['total_common_battery_charge'] = 0
            if self.households[household_id].storage is not None:
                accumulated_household_logs[household_id]['total_local_battery_charge'] = np.sum(self.households[
                                                                                                    household_id].storage.charge)
                accumulated_household_logs[household_id]['total_local_battery_discharge'] = np.sum(self.households[
                                                                                                       household_id].storage.discharge)
            else:
                accumulated_household_logs[household_id]['total_local_battery_charge'] = 0
                accumulated_household_logs[household_id]['total_local_battery_discharge'] = 0
        return accumulated_household_logs

    def append_final_balance(self):
        lock = Lock()
        file_path = self.saving_dir + '/final_balance.csv'
        balance_dict = {}
        for household_id in self.households_ids:
            balance_dict[household_id] = np.sum(self.households[household_id].balance)
        balance_dict['total'] = sum(balance_dict.values())
        # Check if the file exists
        with lock:
            if os.path.exists(file_path):
                # Load the existing CSV file
                df = pd.read_csv(file_path)
                # Ensure all columns in values_dict exist in the DataFrame
                for household_id in self.households_ids:
                    if household_id not in df.columns:
                        df[household_id] = pd.NA
            else:
                # Create a new DataFrame with the columns from values_dict
                df = pd.DataFrame(columns=balance_dict.keys())

            # Append the new row of values
            df = pd.concat([df, pd.DataFrame([balance_dict])], ignore_index=True)
            df = df / self.max_timesteps

            # Save the DataFrame back to the CSV file
            df.to_csv(file_path, index=False)

    @staticmethod
    def save_per_timestep_logs(logs, filename):
        logs_df = pd.DataFrame(logs)
        logs_df.set_index("timestamp", inplace=True)
        logs_df.to_csv(filename, index=True)

    @staticmethod
    def save_per_household_logs(logs, filename):
        df = pd.DataFrame.from_dict(logs, orient='index')
        df.to_csv(filename)

    def save_json_logs(self, logs_dict, name):
        # Save dict to a json file
        with open(name, 'w') as file:
            json.dump(logs_dict, file, indent=4)

    def _log_and_store_episode_end_values(self):
        current_time = self.get_current_time()
        dir_name = self.create_results_directory(current_time)
        timestep_logs = self.calculate_aggregated_values()
        accumulated_household_logs = self.calculate_household_values()

        timestamps = self.households[next(iter(self.households_ids))].gateway.timestamps[:self.max_timesteps]
        timestep_logs['timestamp'] = timestamps

        if self.common_battery is not None:
            timestep_logs['common_battery_charge'] = (self.common_battery.charge_vals[:self.max_timesteps] / self.num_households).tolist()
            timestep_logs['common_battery_discharge'] = (self.common_battery.discharge_vals[:self.max_timesteps] / self.num_households).tolist()
            timestep_logs['common_battery_soc'] = (self.common_battery.soc[:self.max_timesteps]).tolist()
        else:
            timestep_logs['common_battery_charge'] = [0] * self.max_timesteps
            timestep_logs['common_battery_discharge'] = [0] * self.max_timesteps
            timestep_logs['common_battery_soc'] = [0] * self.max_timesteps
        # Print all the fields in timestep logs along with their size
        for key in timestep_logs:
            print(f"{key}: {len(timestep_logs[key])}")

        self.save_per_timestep_logs(timestep_logs, dir_name + '/aggregated_results.csv')
        self.save_per_household_logs(accumulated_household_logs, dir_name + '/accumulated_household_logs.csv')
        df_logs = pd.DataFrame(timestep_logs)
        household = next(iter(self.households.values()))
        if self.auto_plotting:
            plot_energy_logs(df_logs, household.gateway.import_price,
                             household.gateway.export_price, dir_name + '/plot_energy')
            read_and_plot_household_data(dir_name + '/accumulated_household_logs.csv',
                                         save_path=dir_name + '/plot_households')
        return timestep_logs

    def _append_market_info(self):
        self.historic_offers_prices.append(deepcopy(self.local_market_price))
        self.historic_offers_quantities.append(deepcopy(self.current_offers_quantities))

    def print_info(self, info):
        message = f"Environment --- " + info
        if self.printing:
            print(message)
        if self.log_debugging:
            logging.info(message)

    def _calculate_max_timestep(self):
        # Take any element from the households_resources dict
        household_resources = next(iter(self.env_resources.households.values()))
        return household_resources.load.value.shape[0] - self.forecasting_range

    def get_max_timesteps(self):
        return self.max_timesteps


class Household:
    def __init__(self, id: str, resources: HouseholdResource, gateway: Gateway, num_household, pv_efficiency=1.0,
                 num_trading_phases=1, forecasting_step=4, reward_export=False, has_common_battery=False,
                 printing_debug=True, log_debugging=True):
        self.current_timestep = 0

        self.has_common_battery = has_common_battery

        self.resources = deepcopy(resources)

        self.id = id

        self.num_households = num_household

        self.forecasting_range = 16

        self.generator = self.resources.generator
        self.load = self.resources.load
        self.storage = self.resources.storage
        self.gateway = gateway

        if self.generator is not None:
            self.generator.production = self.generator.upper_bound * pv_efficiency
        self.max_timestep = self.gateway.exports.shape[0]

        self.num_trading_phases = num_trading_phases

        self.current_available_energy: float = 0

        self.forecasting_step = forecasting_step

        self._create_observation_space()
        self._create_action_space()

        self.accumulated_market_cost: float = 0.0

        # Track quantities of energy traded with other households, retailer, and produced
        self.local_imports = np.zeros(self.max_timestep)
        self.local_exports = np.zeros(self.max_timestep)
        self.produced = np.zeros(self.max_timestep)
        self.charge_common_battery = np.zeros(self.max_timestep)
        self.discharge_common_battery = np.zeros(self.max_timestep)
        self.balance = np.zeros(self.max_timestep)

        self.household_balance_history = []

        self.printing = printing_debug
        self.log_debugging = log_debugging
        self.reward_export = reward_export

        self.current_charge_reward = 0

        self.exchanged_energy = 0

        self.storage_rewards = np.zeros(self.max_timestep)

    def _create_observation_space(self) -> None:
        self._obs_space_in_preferred_format = True
        temp_observation_space = {
            'current_buy_price': gym.spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32),
            'current_sell_price': gym.spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32),
            'current_loads': gym.spaces.Box(low=0, high=21.0, shape=(1,), dtype=np.float32),
            'future_loads': gym.spaces.Box(low=0, high=21.0, shape=(1,), ),
            'exchanged_energy': gym.spaces.Box(low=-10, high=10, dtype=np.float32)
        }

        if self.storage is not None:
            temp_observation_space.update(
                {'current_soc': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)})

        if self.has_common_battery:
            temp_observation_space.update(
                {'current_common_soc': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)})

        if self.forecasting_step > 0:
            temp_observation_space.update({
                'forecasted_buy_price': gym.spaces.Box(low=-100, high=100.0,
                                                       shape=(int(self.forecasting_range / self.forecasting_step),),
                                                       dtype=np.float32),

            })
            if self.generator is not None:
                temp_observation_space.update({
                    'forecasted_generation': gym.spaces.Box(low=-50, high=50.0,
                                                            shape=(
                                                                int(self.forecasting_range / self.forecasting_step),),
                                                            dtype=np.float32),
                })

        # Set the observation space
        self.observation_space = gym.spaces.Dict(temp_observation_space)

    # Handle Action Space
    def _create_action_space(self) -> None:
        self._action_space_in_preferred_format = True
        temp_action_space = {}

        temp_action_space.update(self._create_storage_actions())

        temp_action_space.update(self._create_market_actions())

        self.action_space = gym.spaces.Dict(temp_action_space)

    def _create_market_actions(self) -> dict:
        if self.num_trading_phases > 0:
            return {
                'trade_action': gym.spaces.Discrete(3),
                'current_offers_quantity': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
            }
        return {}

    def step(self, action: dict) -> tuple:
        cost = self.execute_action(action)

        real_reward = -cost
        info = {}
        return real_reward, info

    def execute_action(self, actions) -> float:
        total_cost = 0

        if self.generator is not None:
            self._generate_energy()

        if self.storage is not None:
            self._execute_storage_actions(actions)

        if self.current_available_energy < 0:
            import_cost = self._execute_gateway_actions()
            total_cost += import_cost

        elif self.current_available_energy > 0:
            export_cost = self._execute_gateway_actions()
            if self.reward_export:
                total_cost += export_cost

        self.print_household_info("Total cost: " + str(total_cost))
        self.household_balance_history.append(self.current_available_energy)

        return total_cost

    def _generate_energy(self):
        # In all cases, the generation is equal to the maximum possible energy generation value
        produced_energy: float = self.generator.production[self.current_timestep]
        self.current_available_energy += produced_energy
        self.produced[self.current_timestep] = produced_energy

        self.print_household_info("Produced energy: " + str(produced_energy))
        self.print_household_info("Current available energy: " + str(self.current_available_energy))

        return produced_energy

    def _create_storage_actions(self) -> dict:
        storage_actions = {}
        if self.storage is not None:
            storage_actions.update({
                'storage_action_type': gym.spaces.Discrete(3),  # 0 -> charge , 1 -> discharge, 2 -> do nothing
                'storage_action_value': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
            })

        if self.has_common_battery:
            storage_actions.update({
                'common_battery_action_type': gym.spaces.Discrete(3),  # 0 -> charge , 1 -> discharge, 2 -> do nothing
                'common_battery_action_value': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
            })

        return storage_actions

    def _execute_storage_actions(self, actions):
        self.storage.soc[self.current_timestep] = self.storage.soc[self.current_timestep - 1]

        energy_balance = -self.load.value[self.current_timestep]
        if self.generator:
            energy_balance += self.generator.production[self.current_timestep]
        # Charge state
        if actions['storage_action_type'] == 0:
            # Percent of the charge_max the household is willing to use at a given moment
            charge = actions['storage_action_value'][0]
            final_charge = charge * self.storage.charge_max * self.storage.charge_efficiency / self.storage.capacity_max

            if self.storage.soc[self.current_timestep] + final_charge > 1.0:
                # Make sure we stay within the bounds
                final_charge = 1.0 - self.storage.soc[self.current_timestep]
            used_energy = final_charge * self.storage.capacity_max / self.storage.charge_efficiency
            self.current_available_energy -= used_energy

            self.storage.soc[self.current_timestep] += final_charge
            self.storage.charge[self.current_timestep] = used_energy
            self.storage.discharge[self.current_timestep] = 0.0

            self.print_household_info(
                f"Charged {charge} to the total value of {self.storage.soc[self.current_timestep]}")
            # Assert that the battery value is larger than in the previous timestep
            assert self.storage.soc[self.current_timestep] >= self.storage.soc[
                self.current_timestep - 1], "Battery value must be larger than the previous timestep"

        # Discharge state
        elif actions['storage_action_type'] == 1:
            discharge = actions['storage_action_value'][0]
            current_battery_charge = self.storage.soc[self.current_timestep] * self.storage.capacity_max
            retrieved_energy = discharge * self.storage.discharge_max
            # Set discharge as a percentage of the maximum discharge allowed
            if current_battery_charge - retrieved_energy < 0.0:
                retrieved_energy = current_battery_charge
                final_discharge = current_battery_charge / self.storage.capacity_max

                self.storage.soc[self.current_timestep] = 0.0
                self.storage.discharge[self.current_timestep] = retrieved_energy
            else:
                final_discharge = retrieved_energy / self.storage.capacity_max
                #  Update soc, charge and discharge values
                self.storage.soc[self.current_timestep] -= final_discharge
                self.storage.discharge[self.current_timestep] = retrieved_energy

            self.storage.charge[self.current_timestep] = 0.0
            self.current_available_energy += retrieved_energy

            self.print_household_info(
                f"Discharged {final_discharge} to the total value of {self.storage.soc[self.current_timestep]}")

            # Assert that the battery value is smaller than in the previous timestep
            assert self.storage.soc[self.current_timestep] <= 1, "Battery value must be smaller or equal to 1"
            assert self.storage.soc[self.current_timestep] >= 0, "Battery value must be larger or equal to 0"
            assert self.storage.soc[self.current_timestep] <= self.storage.soc[
                self.current_timestep - 1], "Battery value must be smaller than the previous timestep"

    def _execute_gateway_actions(self) -> float:
        cost: float = 0.0

        to_import = 0.0
        to_export = 0.0

        # If we still have energy left, there is no need to import extra energy
        if self.current_available_energy > 0:
            # Force to export all available energy
            to_export = self.current_available_energy
            self.print_household_info(f"Exported {to_export} energy")
            # Update the cost of the export
            # The cost is negative because you earn money by exporting energy
            cost = -to_export * self.gateway.export_price[self.current_timestep]
            self.gateway.exported_cost[self.current_timestep] = cost
            self.print_household_info(f"Exported {to_export} energy at price {cost}")

        # Check if there is a deficit of energy that needs to be imported
        if self.current_available_energy < 0:
            # If not, we are forced to import
            to_import = abs(self.current_available_energy)
            self.print_household_info(f"Imported {to_import} energy")

            cost = to_import * self.gateway.import_price[self.current_timestep]
            self.gateway.imported_cost[self.current_timestep] = cost
            self.print_household_info(f"Imported {to_import} energy at price {cost}")

        self.current_available_energy = 0.0
        self.gateway.imports[self.current_timestep] = to_import
        self.gateway.exports[self.current_timestep] = to_export

        self.balance[self.current_timestep] += cost
        return cost

    def get_initial_observations(self) -> dict:
        observations = {
            'current_buy_price': np.array([self.gateway.import_price[0]],
                                          dtype=np.float32),
            'current_sell_price': np.array([self.gateway.export_price[0]],
                                           dtype=np.float32),
            'current_loads': np.array([self.load.value[0]] if not np.isnan(self.load.value[0]) else [0.0],
                                      dtype=np.float32),
            'future_loads': np.array([self.load.value[1]] if not np.isnan(self.load.value[1]) else [0.0],
                                     dtype=np.float32),
            'exchanged_energy': np.array([self.exchanged_energy] if not np.isnan(self.exchanged_energy) else [0.0],
                                         dtype=np.float32),
        }
        if self.storage is not None:
            observations['current_soc'] = np.array([self.storage.initial_charge],
                                                   dtype=np.float32)

        if self.has_common_battery:
            observations['current_common_soc'] = np.array([0.0], dtype=np.float32)

        if self.forecasting_step > 0:
            forecasted_buy_price = [
                np.mean(self.gateway.price_prediction[i:i + self.forecasting_step])
                for i in range(self.current_timestep + 1, self.current_timestep + 1 + self.forecasting_range,
                               self.forecasting_step)
            ]
            observations.update({
                'forecasted_buy_price': np.array(forecasted_buy_price, dtype=np.float32)
            })
            if self.generator is not None:
                forecasted_generation = [
                    np.mean(self.generator.production[i:i + self.forecasting_step])
                    for i in range(self.current_timestep + 1, self.current_timestep + 1 + self.forecasting_range,
                                   self.forecasting_step)
                ]
                observations.update({
                    'forecasted_generation': np.array(forecasted_generation, dtype=np.float32)
                })
        self.assert_no_nan_observations(observations)

        return observations

    def get_next_observations(self) -> dict:
        observations = {}

        if self.current_timestep >= self.load.value.shape[0]:
            return observations

        observations: dict = {
            'current_buy_price': np.array([self.gateway.import_price[self.current_timestep]],
                                          dtype=np.float32),
            'current_sell_price': np.array([self.gateway.export_price[self.current_timestep]],
                                           dtype=np.float32),
            'current_loads': np.array([self.load.value[self.current_timestep]],
                                      dtype=np.float32),
            'future_loads': np.array([self.load.value[self.current_timestep + 1]],
                                     dtype=np.float32),
            'exchanged_energy': np.array([self.exchanged_energy] if not np.isnan(self.exchanged_energy) else [0.0],
                                         dtype=np.float32),
        }
        if self.storage is not None:
            observations.update(
                {'current_soc': np.array([self.storage.soc[self.current_timestep]], dtype=np.float32)})

        if self.forecasting_step > 0:
            forecasted_observations = self.get_forecasted_observations()
            observations.update(forecasted_observations)

        return observations

    def get_forecasted_observations(self):
        forecasted_observations = {}
        forecast_intervals = range(
            self.current_timestep + 1,
            self.current_timestep + 1 + self.forecasting_range,
            self.forecasting_step
        )

        forecasted_buy_price = [
            np.mean(self.gateway.price_prediction[i:i + self.forecasting_step])
            for i in forecast_intervals
        ]
        forecasted_observations['forecasted_buy_price'] = np.array(forecasted_buy_price, dtype=np.float32)

        if self.generator is not None:
            forecasted_generation = [
                np.mean(self.generator.production[i:i + self.forecasting_step])
                for i in forecast_intervals
            ]
            forecasted_observations['forecasted_generation'] = np.array(forecasted_generation, dtype=np.float32)

        return forecasted_observations

    def assert_no_nan_observations(self, observation):
        for key in observation.keys():
            if np.isnan(observation[key]).any():
                logging.error(
                    f"Observation {key} value is NaN: {observation[key]} at timestep {self.current_timestep} for household {self.id}")
                print(
                    f"Observation {key} value is NaN: {observation[key]} at timestep {self.current_timestep} for household {self.id}")

    def print_household_info(self, info):
        message = f"Household {self.id} --- " + info
        if self.printing:
            print(message)
        if self.log_debugging:
            logging.info(message)
