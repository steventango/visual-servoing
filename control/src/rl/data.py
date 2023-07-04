from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def combine_data(experiment: str, data_paths: Iterable[Path], eval_log_path: Path):
    dfs = []
    for data_path in data_paths:
        zipped_data = np.load(data_path)
        data = dict(zipped_data)
        n_timesteps = data["timesteps"].shape[0]
        n_eval_episodes = data["results"].shape[1]
        best_successes = np.max(np.mean(data["successes"], axis=1), axis=0)
        dfs.append(
            pd.DataFrame(
                {
                    "timesteps": np.repeat(data["timesteps"], n_eval_episodes),
                    "results": data["results"].flatten(),
                    "ep_lengths": data["ep_lengths"].flatten(),
                    "successes": data["successes"].flatten(),
                    "best_successes":  np.full(n_timesteps * n_eval_episodes, best_successes),
                    "hidden_size": np.full(n_timesteps * n_eval_episodes, data["hidden_size"]),
                    "depth": np.full(n_timesteps * n_eval_episodes, data["depth"]),
                    "dof": np.full(n_timesteps * n_eval_episodes, data["dof"]),
                    "n_params": np.full(n_timesteps * n_eval_episodes, data["num_params"]),
                    "alg": np.full(n_timesteps * n_eval_episodes, data["alg"]),
                    "learning_rate": np.full(n_timesteps * n_eval_episodes, data["learning_rate"]),
                    "experiment": np.full(n_timesteps * n_eval_episodes, experiment)
                }
            )
        )
        zipped_data.close()
    df = pd.concat(dfs)
    print(df)
    df.to_csv(eval_log_path / "evaluation.csv")
