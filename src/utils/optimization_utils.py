import json
from datetime import datetime

from itertools import product

from src.experiments.experiment_runner import ExperimentRunner


def search_hyperparameters(hyperparameter_space= None, num_gpus=1, num_rollout_workers=4, num_envs_per_worker=2, num_cpus=8,
                           search_name="param_searching_super_large_vals_classic"):
    # Define the hyperparameters to search over dynamically
    if hyperparameter_space is None:
        hyperparameter_space = {
            "train_batch_size": [50000, 100000],
            "sgd_minibatch_size": [2048, 1024, 512, 256],
            "lr": [5e-5],
            "gamma": [0.99],
            "entropy_coeff": [0.01],
            "clip_param": [0.15],
            "num_sgd_iter": [10],
            "model": [
                {"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"},
                {"fcnet_hiddens": [512, 512, 256], "fcnet_activation": "relu"},

            ]
        }

    keys, values = zip(*hyperparameter_space.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    results = []

    for i, params in enumerate(param_combinations):
        training_params = params.copy()

        experiment = ExperimentRunner(
            [142, 142, 142, 142, 142], training_params=training_params, results_dir=search_name,
            pricing_mode="tou", testing_mode="testing",
            num_gpus=num_gpus, num_cpus=num_cpus, num_rollout_workers=num_rollout_workers,
            num_envs_per_worker=num_envs_per_worker, plot_rewards=False, eval_interval=4, pv_efficiency=2.0,
            trading_phases=0)

        run_result = experiment.run_single_experiment("month")
        del run_result["evaluation_data"]
        del run_result["evaluation_rewards"]
        del run_result["training_rewards"]

        run_result.update(params)
        results.append(run_result)

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'results/search_results_{search_name}_{current_time}.json'
        with open(save_path, 'w') as f:
            json.dump(results, f)

        print(f"Run {i + 1} results:", run_result)

    return results