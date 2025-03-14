import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Global parameters (modify these as needed)
ENV_NAMES = ["ALE/Assault-ram-v5"]
USE_REPLAY_OPTIONS = [True, False]
EPSILON_LIST = [0.01, 0.1, 0.5]
LR_LIST = [0.0001, 0.001, 0.01]
NUM_EPISODES = 1000
METHODS = ["q_learning", "expected_sarsa"]


def load_results_from_files(file_paths):
    """
    Load and merge results from multiple pickle files.
    Each pickle file contains a dictionary of experiment results.
    """
    combined_results = {}
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                results = pickle.load(f)
                combined_results.update(results)
        else:
            print(f"File not found: {file_path}")
    return combined_results


def pad_run(run, target_length):
    """
    Pad a run with its last value if its length is less than target_length.
    """
    if len(run) < target_length:
        run = run + [run[-1]] * (target_length - len(run))
    return run


def main(mode):
    # List of PKL files (update paths as needed)
    pkl_files = [
        "results/plot_data2.pkl",
        "results/plot_data2_noreplay.pkl",
    ]

    # Load all results from the specified pickle files
    results_dict = load_results_from_files(pkl_files)

    # Calculate total number of experiment plots (one for each combination)
    total_plots = len(ENV_NAMES) * len(USE_REPLAY_OPTIONS) * len(EPSILON_LIST) * len(LR_LIST)

    if mode == "save":
        # In "save" mode, combine all plots into one large image using subplots

        # Set the number of columns (adjust as needed) and calculate rows
        ncols = 3
        nrows = int(np.ceil(total_plots / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

        plot_index = 0  # To track current subplot index

        # Loop over all parameter combinations
        for env_name in ENV_NAMES:
            for use_replay in USE_REPLAY_OPTIONS:
                for eps in EPSILON_LIST:
                    for lr in LR_LIST:
                        ax = axs[plot_index]
                        # For each method, plot the learning curves with mean and std deviation
                        for method, color, ls in zip(METHODS, ['green', 'red'], ['solid', 'dashed']):
                            all_runs = []
                            # Experiment key format: "envName_useReplay_epsilon_lr_method_seed"
                            prefix = f"{env_name}_{use_replay}_{eps}_{lr}_{method}"
                            for key, run in results_dict.items():
                                if key.startswith(prefix):
                                    run = pad_run(run, NUM_EPISODES)
                                    all_runs.append(np.array(run))
                            if all_runs:
                                all_runs = np.stack(all_runs)
                                mean_rewards = all_runs.mean(axis=0)
                                std_rewards = all_runs.std(axis=0)
                                episodes = np.arange(1, NUM_EPISODES + 1)
                                ax.plot(episodes, mean_rewards, label=f"{method}", color=color, linestyle=ls)
                                ax.fill_between(episodes,
                                                mean_rewards - std_rewards,
                                                mean_rewards + std_rewards,
                                                color=color, alpha=0.2)
                        # Set subplot title and labels
                        replay_str = "With Replay" if use_replay else "Without Replay"
                        ax.set_title(f"{env_name}\n{replay_str} | ε={eps}, lr={lr}")
                        ax.set_xlabel("Episode")
                        ax.set_ylabel("Total Reward")
                        ax.legend()
                        ax.grid(True)
                        plot_index += 1

        # Remove any unused subplots
        for i in range(plot_index, len(axs)):
            fig.delaxes(axs[i])

        plt.tight_layout()
        # Save the combined figure to a file
        save_path = "combined_plots.png"
        plt.savefig(save_path)
        print(f"Combined plot saved as {save_path}")

    elif mode == "plot":
        # In "plot" mode, display individual figures for each parameter configuration
        for env_name in ENV_NAMES:
            for use_replay in USE_REPLAY_OPTIONS:
                for eps in EPSILON_LIST:
                    for lr in LR_LIST:
                        plt.figure(figsize=(10, 6))
                        for method, color, ls in zip(METHODS, ['green', 'red'], ['solid', 'dashed']):
                            all_runs = []
                            prefix = f"{env_name}_{use_replay}_{eps}_{lr}_{method}"
                            for key, run in results_dict.items():
                                if key.startswith(prefix):
                                    run = pad_run(run, NUM_EPISODES)
                                    all_runs.append(np.array(run))
                            if all_runs:
                                all_runs = np.stack(all_runs)
                                mean_rewards = all_runs.mean(axis=0)
                                std_rewards = all_runs.std(axis=0)
                                episodes = np.arange(1, NUM_EPISODES + 1)
                                plt.plot(episodes, mean_rewards, label=f"{method}", color=color, linestyle=ls)
                                plt.fill_between(episodes,
                                                 mean_rewards - std_rewards,
                                                 mean_rewards + std_rewards,
                                                 color=color, alpha=0.2)
                            else:
                                print(f"No data found for {prefix}")
                        replay_str = "With Replay Buffer" if use_replay else "Without Replay Buffer"
                        plt.xlabel("Episode")
                        plt.ylabel("Total Reward")
                        plt.title(f"{env_name} | {replay_str} | ε={eps}, lr={lr}")
                        plt.legend()
                        plt.grid(True)
                        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["plot", "save"], required=True,
                        help="Mode: 'plot' to display figures individually, 'save' to combine all plots into one image and save it")
    args = parser.parse_args()
    main(args.mode)
