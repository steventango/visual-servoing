from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from pandas.api.types import is_numeric_dtype


def plot_best_success_rate(df, x, xlabel, plot_dir):
    plt.figure()
    sns.set()
    ax = sns.lineplot(
        df,
        x=x,
        y="best_successes"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Best Success Rate")
    ax.set_ylim(-.1, 1.1)
    ax.set_title("WAMVisualReach")
    plt.savefig(plot_dir / f"best_successes.{x}.png")


def plot_timesteps(df, y, ylabel, hue, huelabel, plot_dir, dof=None):
    plt.figure()
    sns.set()
    if dof:
        df = df[df['dof'] == dof]
    if is_numeric_dtype(df[hue]):
        norm = plt.Normalize(df[hue].min(), df[hue].max())
    else:
        norm = None
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    ax: plt.Axes = sns.lineplot(
        df,
        x="timesteps",
        y=y,
        hue=hue,
        hue_norm=norm,
        palette="viridis",

    )
    if is_numeric_dtype(df[hue]):
        if legend := ax.get_legend():
            legend.remove()
        ax.figure.colorbar(sm, label=huelabel)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel(ylabel)
    ax.set_title("WAMVisualReach")
    if dof:
        plt.savefig(plot_dir / f"{y}.{hue}.{dof}.timesteps.png")
    else:
        plt.savefig(plot_dir / f"{y}.{hue}.timesteps.png")


def main():
    experiments = ["360"]
    dfs = []
    for experiment in experiments:
        csv_path = Path(f"experiments/{experiment}/data/evaluation.csv")
        plot_dir = Path(f"experiments/{experiment}/plots")
        plot_dir.mkdir(exist_ok=True)
        df = pd.read_csv(csv_path)
        dfs.append(df)
        plot(plot_dir, df)

    df = pd.concat(dfs)
    del df['experiment']
    # df = df.groupby(["hidden_size", "depth"]).mean().reset_index()
    print(df)
    plot_dir = Path("experiments/plots")
    plot_dir.mkdir(exist_ok=True)
    plot(plot_dir, df)


def plot(plot_dir, df):
    HIDDEN_SIZE = ("hidden_size", "Hidden Size")
    DEPTH = ("depth", "Depth")
    NUMBER_OF_PARAMETERS = ("n_params", "Number of parameters")
    ALGORITHM = ("alg", "Algorithm")
    REWARD = ("results", "Reward")
    SUCCESS_RATE = ("successes", "Success Rate")
    LEARNING_RATE = ("learning_rate", "Learning Rate")
    DOFS = [3, 4, 7]
    plot_best_success_rate(df, *HIDDEN_SIZE, plot_dir)
    plot_best_success_rate(df, *NUMBER_OF_PARAMETERS, plot_dir)
    plot_best_success_rate(df, *DEPTH, plot_dir)
    plot_timesteps(df, *REWARD, *HIDDEN_SIZE, plot_dir)
    plot_timesteps(df, *SUCCESS_RATE, *HIDDEN_SIZE, plot_dir)
    plot_timesteps(df, *REWARD, *DEPTH, plot_dir)
    plot_timesteps(df, *SUCCESS_RATE, *DEPTH, plot_dir)
    plot_timesteps(df, *REWARD, *NUMBER_OF_PARAMETERS, plot_dir)
    plot_timesteps(df, *SUCCESS_RATE, *NUMBER_OF_PARAMETERS, plot_dir)
    for dof in DOFS:
        plot_timesteps(df, *REWARD, *ALGORITHM, plot_dir, dof)
        plot_timesteps(df, *REWARD, *LEARNING_RATE, plot_dir, dof)
        plot_timesteps(df, *SUCCESS_RATE, *LEARNING_RATE, plot_dir, dof)


if __name__ == '__main__':
    main()
