import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
from data import combine_data
from tqdm import tqdm
from train import parse_args, train


def get_jsrl_experiment_args(repeats: int, dofs: List[int], common_args: dict):
    alg = "JSRL-UVS-TD3"
    common_args = deepcopy(common_args)
    common_args.alg = alg
    args_list = []
    # policies = ["MultiInputPolicy", "NJMultiInputPolicy"]
    # policies = ["MultiInputPolicy"]
    policies = ["NJMultiInputPolicy"]
    hyperparameters = list(itertools.product(range(repeats), dofs, policies))
    for i, dof, policy in hyperparameters:
        args = deepcopy(common_args)
        args.dof = dof
        args.policy = policy
        name = f"{alg}-{policy.replace('MultiInputPolicy', '')}_{dof}DOF/{i}"
        args = add_paths_to_args(common_args, args, name)
        args_list.append(args)
    return args_list


def add_paths_to_args(common_args, args, name):
    args.model_path = f"{common_args.model_path}/{name}"
    args.eval_log_path = f"{common_args.eval_log_path}/{name}"
    args.tensorboard_log_path = f"{common_args.tensorboard_log_path}/{name}"
    if common_args.final_video_path:
        args.final_video_path = f"{common_args.final_video_path}/{name}"
    if common_args.learning_video_path:
        args.learning_video_path = f"{common_args.learning_video_path}/{name}"
    return args


def get_rrl_experiment_args(repeats: int, dofs: List[int], common_args: dict):
    alg = "RRL-UVS-TD3"
    common_args = deepcopy(common_args)
    common_args.alg = alg
    args_list = []
    # policies = ["MultiInputPolicy", "NJMultiInputPolicy"]
    policies = ["MultiInputPolicy"]
    hyperparameters = list(itertools.product(range(repeats), dofs, policies))
    for i, dof, policy in hyperparameters:
        args = deepcopy(common_args)
        args.dof = dof
        args.policy = policy
        name = f"{alg}-{policy.replace('MultiInputPolicy', '')}_{dof}DOF/{i}"
        args = add_paths_to_args(common_args, args, name)
        args_list.append(args)
    return args_list


def get_rl_experiment_args(repeats: int, dofs: List[int], common_args: dict):
    alg = "TD3"
    common_args = deepcopy(common_args)
    common_args.alg = alg
    args_list = []
    hyperparameters = list(itertools.product(range(repeats), dofs))
    for i, dof, in hyperparameters:
        args = deepcopy(common_args)
        args.dof = dof
        name = f"{alg}_{dof}DOF/{i}"
        args = add_paths_to_args(common_args, args, name)
        args_list.append(args)
    return args_list


def get_nj_experiment_args(repeats: int, dofs: List[int], common_args: dict):
    alg = "TD3"
    common_args = deepcopy(common_args)
    common_args.alg = alg
    policies = ["RNJMultiInputPolicy"]
    args_list = []
    hyperparameters = list(itertools.product(range(repeats), dofs, policies))
    for i, dof, policy in hyperparameters:
        args = deepcopy(common_args)
        args.dof = dof
        args.policy = policy
        name = f"{alg}_NJ_{policy}_{dof}DOF/{i}"
        args = add_paths_to_args(common_args, args, name)
        args_list.append(args)
    return args_list


def get_uvs_experiment_args(repeats: int, dofs: List[int], common_args: dict):
    alg = "UVS"
    common_args = deepcopy(common_args)
    common_args.alg = alg
    common_args.std = 0.
    args_list = []
    # lrs = [0, 0.01, 0.1, 1]
    lrs = [0]
    hyperparameters = list(itertools.product(range(repeats), dofs, lrs))
    for i, dof, lr in hyperparameters:
        args = deepcopy(common_args)
        args.dof = dof
        args.learning_rate = lr
        name = f"{alg}_{dof}DOF_{lr}LR/{i}"
        args = add_paths_to_args(common_args, args, name)
        args_list.append(args)
    return args_list


def main():
    experiment = "Aug17"
    model_path = Path(f"experiments/{experiment}/models")
    eval_log_path = Path(f"experiments/{experiment}/data")
    final_video_path = Path(f"experiments/{experiment}/videos")
    tensorboard_log_path = Path(f"experiments/{experiment}/logs")
    repeats = 1
    common_args = parse_args(
        [
            "--model_path",
            str(model_path),
            "--eval_log_path",
            str(eval_log_path),
            "--tensorboard_log_path",
            str(tensorboard_log_path),
            "--final_video_path",
            str(final_video_path),
            "--learning_video_path",
            str(final_video_path),
            "--verbose",
            "0",
            "--no_progress_bar",
            "--total_timesteps",
            "100000",
            "--n_envs",
            "1",
            "--hidden_size",
            "16",
            "--depth",
            "2",
        ]
    )
    dofs = [3, 4, 7]
    data_paths = []
    args_list = []
    # args_list += get_uvs_experiment_args(repeats, dofs, common_args)
    # args_list += get_jsrl_experiment_args(repeats, dofs, common_args)
    args_list += get_rrl_experiment_args(repeats, dofs, common_args)
    # args_list += get_rl_experiment_args(repeats, dofs, common_args)
    # args_list += get_nj_experiment_args(repeats, dofs, common_args)

    with ProcessPoolExecutor() as executor:
        futures = {}
        for args in args_list:
            future = executor.submit(train, args)
            futures[future] = args
            data_paths.append(Path(args.eval_log_path) / "evaluations.npz")

        for future in tqdm(as_completed(futures), total=len(futures)):
            num_params = future.result()
            args = futures[future]
            _eval_log_path = Path(args.eval_log_path) / "evaluations.npz"
            data = np.load(_eval_log_path, allow_pickle=True)
            data = dict(data)
            data["hidden_size"] = args.hidden_size
            data["depth"] = args.depth
            data["num_params"] = num_params
            data["dof"] = args.dof
            data["alg"] = args.alg
            data["learning_rate"] = args.learning_rate
            np.savez(_eval_log_path, **data)
            data_paths.append(_eval_log_path)

    combine_data(experiment, data_paths, eval_log_path)


if __name__ == "__main__":
    main()
