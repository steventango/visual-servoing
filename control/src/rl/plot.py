from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm


def plot_best_success_rate(df):
    plt.figure()
    sns.set()
    ax = sns.lineplot(
        df,
        x="hidden_size",
        y="best_successes"
    )
    ax.set_xlabel("Hidden Size")
    ax.set_ylabel("Best Success Rate")
    ax.set_title("WAMVisualReach")
    plt.savefig("plots/best_successes.hidden_size.png")


def plot_timesteps(df, y, ylabel):
    plt.figure()
    sns.set()
    norm = plt.Normalize(df["hidden_size"].min(), df["hidden_size"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    ax: plt.Axes = sns.lineplot(
        df,
        x="timesteps",
        y=y,
        hue="hidden_size",
        hue_norm=norm,
        palette="viridis",

    )
    ax.get_legend().remove()
    ax.figure.colorbar(sm, label="Hidden Size")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel(ylabel)
    ax.set_title("WAMVisualReach")
    plt.savefig(f"plots/{y}.timesteps.png")


def main():
    data_path = Path("data")

    data = np.load(data_path / f"control/src/rl/models/experiment_{1}/evaluations.npz")
    data = dict(data)
    print({k: v.shape for k, v in data.items()})

    n = 256

    dfs = []
    for i in tqdm(range(1, n + 1)):
        zipped_data = np.load(data_path / f"control/src/rl/models/experiment_{i}/evaluations.npz")
        data = dict(zipped_data)
        n_timesteps = data["timesteps"].shape[0]
        n_eval_episodes = data["results"].shape[1]
        best_successes = np.max(data["successes"], axis=0)
        dfs.append(
            pd.DataFrame(
                {
                    "timesteps": np.repeat(data["timesteps"], n_eval_episodes),
                    "results": data["results"].flatten(),
                    "ep_lengths": data["ep_lengths"].flatten(),
                    "successes": data["successes"].flatten(),
                    "best_successes": np.repeat(best_successes, n_timesteps),
                    "hidden_size": np.full(n_timesteps * n_eval_episodes, data["hidden_size"]),
                    "n_params": np.full(n_timesteps * n_eval_episodes, data["num_params"])
                }
            )
        )
        zipped_data.close()
    df = pd.concat(dfs)
    print(df)
    # plot_best_success_rate(df)
    plot_timesteps(df, "results", "Reward")
    plot_timesteps(df, "successes", "Success Rate")


if __name__ == '__main__':
    main()
