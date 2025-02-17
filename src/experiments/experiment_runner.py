import json
import time
import warnings
import os
import random
import logging
from collections import defaultdict
from pathlib import Path
from itertools import groupby
from typing import List, Dict, Union

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

from src.environments.lec_resources import EnvironmentResources
from src.parsers.pecan_parser import PecanParser
from src.environments.lec_environment import LECEnvironment
from src.utils.plotting.reward_plotter import RewardPlotter


class ExperimentRunner:
    def __init__(
            self,
            household_ids: List[int],
            trading_phases: int = 1,
            training_params: Dict = None,
            results_dir: str = None,
            pricing_mode: str = "tou",
            plot_rewards: bool = False,
            eval_interval: int = 32,
            pv_efficiency: float = 1.0,
            testing_mode: str = "testing",
            num_gpus: int = 1,
            num_cpus: int = 16,
            num_rollout_workers: int = 15,
            num_envs_per_worker: int = 2,
            log_plotting: bool = False,
            reward_export: bool = True,
            common_reward_factor: float = 0.0,
            price_based_common_reward_factor: bool = False,
            common_battery_type: str = "none"
    ):
        self.household_ids = household_ids
        self.trading_phases = trading_phases
        self.training_params = training_params or {
            "lr": 5e-5,
            "gamma": 0.99,
            "train_batch_size": 50000,
            "sgd_minibatch_size": 512,
            "clip_param": 0.15,
            "num_sgd_iter": 10,
            'model': {
                'fcnet_hiddens': [256, 256],
                'fcnet_activation': 'relu',
            },
        }
        self.results_dir = results_dir or time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.plot_rewards = plot_rewards
        self.pricing_mode = pricing_mode
        self.eval_interval = eval_interval
        self.pv_efficiency = pv_efficiency
        self.testing_mode = testing_mode
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.num_rollout_workers = num_rollout_workers
        self.num_envs_per_worker = num_envs_per_worker
        self.log_plotting = log_plotting
        self.reward_export = reward_export
        self.common_battery_type = common_battery_type
        self.common_reward_factor = common_reward_factor
        self.price_based_common_reward_factor = price_based_common_reward_factor

        self.current_iteration = 0
        # For battery efficiency, note: local_battery_efficiency is set via setter later.
        self.local_battery_efficiency = None

    def set_household_ids(self, household_ids: List[int]):
        self.household_ids = household_ids

    def set_results_dir(self, results_dir: str):
        self.results_dir = results_dir

    def set_pv_efficiency(self, new_pv_efficiency: float):
        self.pv_efficiency = new_pv_efficiency

    def set_battery_efficiency(self, battery_efficiency: float):
        self.local_battery_efficiency = battery_efficiency

    def get_parsed_resources(self, scenario: str) -> EnvironmentResources:
        parser = PecanParser(
            'data/pecan_street/15minute_data_newyork.csv',
            'data/pecan_street/metadata.csv',
            '',
            pricing_mode=self.pricing_mode,
            common_battery_type=self.common_battery_type
        )
        parser.parse()
        return parser.get_preset_resources(scenario, self.household_ids)

    def get_next_run_results_path(self, scenario: str) -> (str, str):
        i = 1
        train_pattern = os.path.join(self.results_dir, "train", f"{scenario}_run_results_{str(i).zfill(2)}")
        while os.path.exists(train_pattern):
            i += 1
            train_pattern = os.path.join(self.results_dir, "train", f"{scenario}_run_results_{str(i).zfill(2)}")
        train_run_dir = train_pattern
        test_run_dir = os.path.join(self.results_dir, "test", f"{scenario}_run_results_{str(i).zfill(2)}")
        return train_run_dir, test_run_dir

    @staticmethod
    def reduce_to_distinct(rewards: List[Union[float, None]]) -> List[Union[float, None]]:
        return [key for key, _ in groupby(rewards)]

    @staticmethod
    def check_reward_plateau(rewards: List[Union[float, None]], window_size: int = 5, threshold: float = 0.1) -> bool:
        reduced = ExperimentRunner.reduce_to_distinct(rewards)
        if len(reduced) <= window_size + 1:
            return False
        recent = [r for r in reduced[-window_size:] if r is not None]
        return (max(recent) - min(recent)) < threshold

    @staticmethod
    def check_reward_no_improvement(rewards: List[Union[float, None]], patience: int = 5,
                                    threshold: float = 0.1) -> bool:
        reduced = ExperimentRunner.reduce_to_distinct(rewards)
        if len(reduced) <= patience + 1:
            return False
        best = max(r for r in reduced[1:-patience] if r is not None)
        recent = reduced[-patience:]
        return all(r is not None and r < best + threshold for r in recent)

    @staticmethod
    def check_stop_condition(iteration: int, eval_rewards: List[float], train_rewards: List[float]) -> bool:
        return iteration > 50 and (
                ExperimentRunner.check_reward_no_improvement(train_rewards) or
                ExperimentRunner.check_reward_plateau(train_rewards)
        )

    @staticmethod
    def assign_policies(env) -> Dict:
        policies = {}
        for household_id in env.households:
            policies[str(household_id)] = (
                None,
                env.observation_space[household_id],
                env.action_space[household_id],
                {}
            )
        return policies

    def register_envs(self, scenario: str, run_name: str, eval_dir: str = None) -> (LECEnvironment, LECEnvironment):
        train_resources = self.get_parsed_resources(scenario)

        test_resources = self.get_parsed_resources(self.testing_mode)

        train_run_dir, test_run_dir = self.get_next_run_results_path(run_name)
        if eval_dir:
            test_run_dir = eval_dir

        run_env = LECEnvironment(
            env_resources=train_resources,
            saving_dir=train_run_dir,
            trading_phases=self.trading_phases,
            printing=False,
            pv_efficiency=self.pv_efficiency,
            auto_plotting=False,
            log_debugging=False,
            reward_export=self.reward_export,
            common_reward_factor=self.common_reward_factor,
            price_based_common_reward_factor=self.price_based_common_reward_factor
        )

        test_env = LECEnvironment(
            env_resources=test_resources,
            saving_dir=test_run_dir,
            trading_phases=self.trading_phases,
            printing=False,
            pv_efficiency=self.pv_efficiency,
            auto_plotting=self.log_plotting,
            log_debugging=False,
            reward_export=True
        )

        register_env("EC_Multi", lambda config: run_env)
        register_env("EC_Multi_Test", lambda config: test_env)
        return run_env, test_env

    def run_single_experiment(self, scenario: str, run_name: str = None) -> Dict:
        ray.shutdown()
        ray.init(include_dashboard=True, ignore_reinit_error=True, num_cpus=self.num_cpus, num_gpus=self.num_gpus)

        parser = PecanParser(
            'data/pecan_street/15minute_data_newyork.csv',
            'data/pecan_street/metadata.csv',
            '',
            pricing_mode=self.pricing_mode,
            common_battery_type=self.common_battery_type
        )
        parser.parse()
        run_name = run_name or scenario

        train_run_dir, test_run_dir = self.get_next_run_results_path(run_name)

        run_env, test_env = self.register_envs(scenario, run_name)
        print("Total timesteps per episode:", run_env.get_max_timesteps())
        policies = ExperimentRunner.assign_policies(run_env)

        # Dynamically set train_batch_size based on environment timesteps if not provided
        if "train_batch_size" not in self.training_params:
            self.training_params["train_batch_size"] = (
                    run_env.get_max_timesteps() * self.num_envs_per_worker * self.num_rollout_workers
            )
            print("Setting train_batch_size to", self.training_params["train_batch_size"])

        test_config = (
            PPOConfig()
            .environment(env="EC_Multi_Test")
            .resources(num_gpus=self.num_gpus)
            .training(**self.training_params)
            .framework("torch")
            .env_runners(num_env_runners=8, rollout_fragment_length="auto", sample_timeout_s=1e6)
            .reporting()
            .multi_agent(
                policies=policies,
                policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: str(agent_id)
            )
            .evaluation(evaluation_interval=1, evaluation_duration_unit="episodes")
        )
        test_config["observation_filter"] = "MeanStdFilter"

        config = (
            PPOConfig()
            .environment(env="EC_Multi")
            .training(**self.training_params)
            .resources(num_gpus=self.num_gpus)
            .framework("torch")
            .env_runners(
                num_rollout_workers=self.num_rollout_workers,
                num_envs_per_worker=self.num_envs_per_worker,
                rollout_fragment_length="auto",
                sample_timeout_s=1e6
            )
            .reporting()
            .multi_agent(
                policies=policies,
                policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: str(agent_id)
            )
            .evaluation(
                evaluation_config=test_config,
                evaluation_interval=self.eval_interval,
                evaluation_num_env_runners=1,
                evaluation_duration_unit="episodes",
                evaluation_duration=3,
            )
        )
        config["observation_filter"] = "MeanStdFilter"

        formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        log_file = f"results/logs/training_info_logs_{formatted_time}.log"
        logging.basicConfig(
            filename=log_file, level=logging.INFO, filemode='w',
            format="%(asctime)s - %(message)s"
        )

        algo = config.build()

        training_rewards = []
        training_rewards_per_ts = []
        training_rewards_stds_per_ts = []
        evaluation_rewards = [None]
        evaluation_rewards_per_ts = [None]
        evaluation_rewards_stds_per_ts = [None]
        evaluation_data = []
        best_evaluation_reward = float("-inf")
        best_evaluation_rewards_per_ts_per_hh = None
        best_checkpoint_dir = None

        _, baseline_train_cost = parser.calculate_baseline_costs_with_solar(
            household_ids=self.household_ids,
            pv_efficiency=self.pv_efficiency,
            scenario=scenario,
            num_timesteps=run_env.get_max_timesteps()
        )
        _, baseline_test_cost = parser.calculate_baseline_costs_with_solar(
            household_ids=self.household_ids,
            pv_efficiency=self.pv_efficiency,
            scenario=self.testing_mode,
            num_timesteps=test_env.get_max_timesteps()
        )
        baseline_test_reward_per_timestep = -baseline_test_cost / test_env.get_max_timesteps()
        print("Baseline test reward per timestep:", baseline_test_reward_per_timestep)
        baseline_train_reward_per_timestep = -baseline_train_cost / run_env.get_max_timesteps()

        plotter = (
            RewardPlotter(
                self.eval_interval,
                baseline_train_reward=baseline_train_reward_per_timestep,
                baseline_test_reward=baseline_test_reward_per_timestep,
                l_curve_save_path=os.path.join(train_run_dir, "rewards_plot.png")
            )
            if self.plot_rewards else None
        )

        stop_flag = False
        training_iteration = 0
        while not stop_flag:
            os.makedirs(os.path.join(train_run_dir, str(training_iteration)), exist_ok=True)
            os.makedirs(os.path.join(test_run_dir, str(training_iteration)), exist_ok=True)

            results = algo.train()
            training_reward = results['env_runners']['episode_reward_mean'] / run_env.num_households
            training_reward *= self.num_envs_per_worker  # Adjust for rollout scaling
            training_rewards.append(training_reward if not np.isnan(training_reward) else None)
            training_reward_per_ts = training_reward / run_env.get_max_timesteps() if training_reward and not np.isnan(
                training_reward) else None
            training_rewards_per_ts.append(training_reward_per_ts)
            train_rewards_per_hh = results['env_runners']['policy_reward_mean']
            print("Training rewards per household:", train_rewards_per_hh)
            std_reward = np.std([
                (value * self.num_envs_per_worker / run_env.get_max_timesteps()) if value and not np.isnan(value) else 0
                for value in train_rewards_per_hh.values()
            ])
            training_rewards_stds_per_ts.append(std_reward)

            print(
                f"Iter: {training_iteration}; avg. training reward={training_reward} avg. reward per timestep={training_reward_per_ts}")
            logging.info(
                f"Iter: {training_iteration}; avg. training reward={training_reward} avg. reward per timestep={training_reward_per_ts}")

            if training_iteration > 0 and "evaluation" in results:
                evaluation_metrics = results["evaluation"]
                evaluation_data.append(evaluation_metrics)
                avg_eval_reward = evaluation_metrics['env_runners']["episode_reward_mean"] / test_env.num_households
                evaluation_rewards.append(avg_eval_reward)
                evaluation_reward_per_ts = avg_eval_reward / test_env.get_max_timesteps()
                evaluation_rewards_per_ts.append(evaluation_reward_per_ts)
                eval_reward_per_hh = evaluation_metrics['env_runners']['policy_reward_mean']
                eval_reward_per_hh_per_ts = {
                    key: (value / test_env.get_max_timesteps()) if value and not np.isnan(value) else 0
                    for key, value in eval_reward_per_hh.items()
                }
                std_eval = np.std(list(eval_reward_per_hh_per_ts.values()))
                evaluation_rewards_stds_per_ts.append(std_eval)

                print(
                    f"Iter: {training_iteration}; avg. evaluation reward={avg_eval_reward} avg. reward per timestep={evaluation_reward_per_ts}")
                logging.info(
                    f"Iter: {training_iteration}; avg. evaluation reward={avg_eval_reward} avg. reward per timestep={evaluation_reward_per_ts}")

                checkpoint_dir = os.path.join(train_run_dir, "checkpoints",
                                              f"checkpoint_{str(training_iteration).zfill(2)}")
                algo.save(checkpoint_dir)
                print("Checkpoint saved at", checkpoint_dir)
                logging.info(f"Checkpoint saved at {checkpoint_dir}")

                if evaluation_rewards[-1] > best_evaluation_reward:
                    best_evaluation_reward = evaluation_rewards[-1]
                    best_evaluation_rewards_per_ts_per_hh = eval_reward_per_hh_per_ts
                    best_checkpoint_dir = checkpoint_dir

                if ExperimentRunner.check_stop_condition(training_iteration, evaluation_rewards_per_ts,
                                                         training_rewards_per_ts):
                    stop_flag = True

            # Clean up empty directories
            for base_dir in (train_run_dir, test_run_dir):
                iter_dir = os.path.join(base_dir, str(training_iteration))
                try:
                    if not os.listdir(iter_dir):
                        os.rmdir(iter_dir)
                except OSError as e:
                    warnings.warn(f"Warning: Failed to delete directory {iter_dir}: {e}")

            if self.plot_rewards and plotter:
                plotter.update_plot(training_rewards_per_ts, training_rewards_stds_per_ts, evaluation_rewards_per_ts,
                                    evaluation_rewards_stds_per_ts)

            training_iteration += 1

        if self.plot_rewards and plotter:
            plotter.finalize()

        env_params = {
            "common_battery_type": self.common_battery_type,
            "trading_phases": self.trading_phases,
            "battery_efficiency": self.local_battery_efficiency,
            "pv_efficiency": self.pv_efficiency,
            "reward_export": self.reward_export,
            "common_reward_factor": self.common_reward_factor,
        }

        experiments_results = {
            "training_rewards": training_rewards,
            "training_rewards_per_ts": training_rewards_per_ts,
            "training_rewards_stds": training_rewards_stds_per_ts,
            "evaluation_rewards": evaluation_rewards,
            "evaluation_rewards_per_ts": evaluation_rewards_per_ts,
            "evaluation_rewards_stds": evaluation_rewards_stds_per_ts,
            "evaluation_data": evaluation_data,
            "log_file": log_file,
            "train_run_dir": train_run_dir,
            "test_run_dir": test_run_dir,
            "scenario": scenario,
            "best_checkpoint_dir": best_checkpoint_dir,
            "best_evaluation_reward": best_evaluation_reward,
            "best_evaluation_rewards_per_ts_per_hh": best_evaluation_rewards_per_ts_per_hh,
            "baseline_train_reward_per_timestep": baseline_train_reward_per_timestep,
            "baseline_test_reward_per_timestep": baseline_test_reward_per_timestep,
            "num_training_timesteps": run_env.get_max_timesteps(),
            "num_testing_timesteps": test_env.get_max_timesteps(),
            "env_params": env_params,
            "household_ids": self.household_ids,
        }

        with open(os.path.join(train_run_dir, "results.json"), 'w') as f:
            json.dump(experiments_results, f)

        ray.shutdown()
        return experiments_results

    def run_multiple_training_runs(self, scenarios: Union[List[str], Dict[str, int]], run_name: str = None) -> List[
        Dict]:
        """Run experiments for multiple scenarios."""
        results = []
        if isinstance(scenarios, dict):
            scenarios = [scenario for scenario, count in scenarios.items() for _ in range(count)]
        for scenario in scenarios:
            results.append(self.run_single_experiment(scenario, run_name))
        return results

    def run_matches(self, num_matches: int = 10, match_results_filename: str = None,
                    limited_env_name: str = None, rich_env_name: str = None) -> List[Dict]:
        limited_env_name = limited_env_name or "scarce"
        rich_env_name = rich_env_name or "abundant"
        match_results_filename = match_results_filename or "match_results"

        ray.shutdown()
        ray.init(include_dashboard=True, ignore_reinit_error=True, num_cpus=self.num_cpus, num_gpus=self.num_gpus)

        weights_map = self.extract_weights_dirs_from_results()
        self.register_envs("few_months", limited_env_name)
        self.register_envs("few_months", rich_env_name)
        match_results = []
        match_results_path = os.path.join(self.results_dir, f"{match_results_filename}.json")
        if os.path.exists(match_results_path):
            print("Found existing match results file. Loading...")
            with open(match_results_path, 'r') as file:
                match_results = json.load(file)

        for i in range(num_matches):
            print(f"Running match {i}")
            first_policy_pair = (
            random.choice(weights_map[rich_env_name]), random.choice(weights_map[limited_env_name]))
            second_policy_pair = (
            random.choice(weights_map[rich_env_name]), random.choice(weights_map[limited_env_name]))

            single_results = {
                "first_policy_pair": first_policy_pair,
                "second_policy_pair": second_policy_pair,
                "results": self.run_full_match(first_policy_pair, second_policy_pair)
            }
            match_results.append(single_results)
            with open(match_results_path, 'w') as f:
                json.dump(match_results, f)
        return match_results

    def run_full_match(self, policy_pair_1, policy_pair_2) -> Dict:
        results = {
            "abundant": self.run_single_match(policy_pair_1[0], policy_pair_2[0], "abundant"),
            "scarce": self.run_single_match(policy_pair_1[1], policy_pair_2[1], "scarce"),
            "mixed1": self.run_single_match(policy_pair_1[0], policy_pair_2[1], "mixed1"),
            "mixed2": self.run_single_match(policy_pair_1[1], policy_pair_2[0], "mixed2"),
        }
        return results

    def aggregate_weights(self, weights: Dict) -> Dict:
        aggregated = defaultdict(list)
        for household_id, weight in weights.items():
            household_number = int(household_id.split('_')[0])
            aggregated[household_number].append(weight)
        return aggregated

    def run_single_match(self, checkpoint_path1: str, checkpoint_path2: str, match_name: str) -> Dict:
        # Load weights from checkpoints
        algo1 = Algorithm.from_checkpoint(checkpoint_path1)
        weights1 = algo1.get_weights()
        algo1.stop()

        algo2 = Algorithm.from_checkpoint(checkpoint_path2)
        weights2 = algo2.get_weights()
        algo2.stop()

        weights1_agg = self.aggregate_weights(weights1)
        weights2_agg = self.aggregate_weights(weights2)

        household_ids1 = list(weights1_agg.keys())
        household_ids2 = list(weights2_agg.keys())
        if household_ids1 != household_ids2:
            raise ValueError("Household ids do not match between algorithms.")

        new_weights = {}
        if len(weights1_agg.keys()) == 1:
            household_id = list(weights1_agg.keys())[0]
            new_weights[f"{household_id}_0"] = random.choice(weights1_agg[household_id])
            new_weights[f"{household_id}_1"] = random.choice(weights2_agg[household_id])
        elif len(weights1_agg.keys()) == 2:
            household_id1, household_id2 = household_ids1[0], household_ids1[1]
            new_weights[f"{household_id1}_0"] = random.choice(weights1_agg[household_id1])
            new_weights[f"{household_id2}_1"] = random.choice(weights2_agg[household_id2])
        else:
            raise ValueError("Matching not supported for more than 2 households.")

        parser = PecanParser(
            'data/pecan_street/15minute_data_newyork.csv',
            'data/pecan_street/metadata.csv',
            '',
            pricing_mode=self.pricing_mode
        )
        parser.parse()
        household_ids = [int(s.split('_')[0]) for s in new_weights.keys()]
        matching_resources = parser.get_preset_resources("testing", household_ids)

        k = 0
        while os.path.exists(os.path.join("results", "matches", f"match_run_results_{str(k).zfill(2)}")):
            k += 1

        match_run_dir = os.path.join("results", "matches", self.results_dir, match_name,
                                     f"match_run_results_{str(k).zfill(2)}")
        matching_env = LECEnvironment(
            env_resources=matching_resources,
            saving_dir=match_run_dir,
            printing=False,
            trading_phases=self.trading_phases,
            pv_efficiency=self.pv_efficiency,
            auto_plotting=False,
            reward_export=self.reward_export,
            log_debugging=False
        )
        register_env("EC_Multi_Match", lambda config: matching_env)
        obs_space = matching_env.observation_space
        act_space = matching_env.action_space

        policies = {
            str(agent_id): PolicySpec(
                policy_class=None,
                observation_space=obs_space[agent_id],
                action_space=act_space[agent_id],
                config={}
            ) for agent_id in matching_env.households_ids
        }

        match_config = (
            PPOConfig()
            .environment(env="EC_Multi_Match")
            .resources(num_gpus=self.num_gpus)
            .training(**self.training_params)
            .framework("torch")
            .env_runners(num_env_runners=7, num_envs_per_worker=1, rollout_fragment_length="auto", sample_timeout_s=1e6)
            .reporting()
            .multi_agent(policies=policies, policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: str(agent_id))
            .evaluation(evaluation_interval=1, evaluation_duration_unit="episodes")
        )
        match_config["observation_filter"] = "MeanStdFilter"

        config = (
            PPOConfig()
            .environment(env="EC_Multi")
            .training(**self.training_params)
            .resources(num_gpus=self.num_gpus)
            .framework("torch")
            .env_runners(num_rollout_workers=7, num_envs_per_worker=1, rollout_fragment_length="auto",
                         sample_timeout_s=1e6)
            .reporting()
            .multi_agent(policies=policies, policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: str(agent_id))
            .evaluation(evaluation_config=match_config, evaluation_interval=self.eval_interval,
                        evaluation_num_env_runners=1, evaluation_duration_unit="episodes", evaluation_duration=5)
        )
        config["observation_filter"] = "MeanStdFilter"

        algo = config.build()
        algo.set_weights(new_weights)
        results = algo.evaluate()
        print("Evaluation results:", results)
        return results

    def extract_best_rewards_from_results(self) -> Dict:
        results_dict = os.path.join(self.results_dir, "train")
        best_reward = defaultdict(list)
        for dirpath, dirnames, _ in os.walk(results_dict):
            for dirname in dirnames:
                env_type = dirname.split('_')[0]
                file_path = os.path.join(dirpath, dirname, "results.json")
                with open(file_path) as f:
                    best_reward[env_type].append(json.load(f)["best_evaluation_reward"])
            break
        return best_reward

    def extract_weights_dirs_from_results(self) -> Dict:
        weights_map = defaultdict(list)
        results_dict = os.path.join(self.results_dir, "train")
        for dirpath, dirnames, _ in os.walk(results_dict):
            for dirname in dirnames:
                env_type = dirname.split('_')[0]
                results_file = os.path.join(dirpath, dirname, "results.json")
                try:
                    with open(results_file) as file:
                        best_checkpoint_dir = json.load(file)['best_checkpoint_dir']
                    weights_map[env_type].append(best_checkpoint_dir)
                    print("Extracted weights from", best_checkpoint_dir)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {results_file}: {e}. Skipping...")
            break
        print("Final weights map:", weights_map)
        return weights_map

    @staticmethod
    def extract_run_results_dir_path(path: str) -> Union[str, None]:
        path_obj = Path(path)
        for i, part in enumerate(path_obj.parts):
            if "_run_results" in part:
                return str(Path(*path_obj.parts[: i + 1]))
        return None
