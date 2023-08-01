import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np
from data import combine_data
from tqdm import tqdm
from train import parse_args, train


def main():
    experiment = "JSRL_3"
    model_path = Path(f"experiments/{experiment}/models")
    eval_log_path = Path(f"experiments/{experiment}/data")
    tensorboard_log_path = Path(f"experiments/{experiment}/logs")
    repeats = 1
    args = parse_args(
        [
            "--model_path",
            str(model_path),
            "--eval_log_path",
            str(eval_log_path),
            "--tensorboard_log_path",
            str(tensorboard_log_path),
            "--verbose",
            "0",
            # "--no_progress_bar",
            "--total_timesteps",
            "100000",
            "--n_envs",
            "1",
            # "--std",
            # "0"
        ]
    )

    hidden_sizes = [32]
    depths = [2]
    dofs = [3, 4, 7]
    algs = ["JSRL-UVS-TD3", "TD3", "UVS"]
    # learning_rates = [0, 1]
    # learning_rates = [2, 3, 6, 10]
    # learning_rates = [0.2, 0.4, 0.6, 0.8]
    hyperparameters = list(itertools.product(dofs, algs))
    data_paths = []

    with ProcessPoolExecutor() as executor:
        futures = {}
        for dof, alg in hyperparameters:
            for i in range(repeats):
                args = deepcopy(args)
                args.dof = dof
                args.alg = alg
                # args.hidden_size = hidden_sizes[0]
                # args.depth = depths[0]
                name = f"{alg}_{dof}DOF/{i}"
                args.model_path = str(model_path / name)
                args.eval_log_path = str(eval_log_path / name)
                data_paths.append(Path(args.eval_log_path) / "evaluations.npz")
                args.tensorboard_log_path = str(tensorboard_log_path / name)
                future = executor.submit(train, args)
                futures[future] = args

        for future in tqdm(as_completed(futures), total=len(futures)):
            num_params = future.result()
            args = futures[future]
            _eval_log_path = Path(args.eval_log_path) / "evaluations.npz"
            data = np.load(_eval_log_path)
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
