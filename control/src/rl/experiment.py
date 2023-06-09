from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np
from tqdm import tqdm
from train import parse_args, train


def main():
    args = parse_args(
        [
            "control/src/rl/models/experiment",
            "--verbose",
            "0",
            "--no_progress_bar",
        ]
    )
    data_path = Path("data")
    data_path.mkdir(exist_ok=True)

    hidden_sizes = np.arange(1, 256 + 1, 1)

    with ProcessPoolExecutor() as executor:
        futures = {}
        for i, hidden_size in enumerate(hidden_sizes):
            args = deepcopy(args)
            args.hidden_size = hidden_size
            future = executor.submit(train, args)
            futures[future] = i

        for future in tqdm(as_completed(futures), total=len(futures)):
            num_params = future.result()
            i = futures[future]
            hidden_size = hidden_sizes[i]
            evaluations_log_path = (
                str(data_path / args.model_path)
                + f"_{hidden_size}/evaluations.npz"
            )
            data = np.load(evaluations_log_path)
            data = dict(data)
            data["hidden_size"] = hidden_size
            data["num_params"] = num_params
            np.savez(evaluations_log_path, **data)


if __name__ == "__main__":
    main()
