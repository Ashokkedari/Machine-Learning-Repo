"""
Visualize Q-learning on FrozenLake.

Run: python visualize.py
Needs: pip install "gymnasium[toy-text]" matplotlib
(model.py must be in the same folder)
"""
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from model import (train_q_learning, extract_greedy_policy, run_greedy_episode,
                   evaluate_success_rate, greedy_action, max_q_value)

# Action meanings in FrozenLake: 0=Left, 1=Down, 2=Right, 3=Up
ARROWS = {0: (-0.3, 0), 1: (0, 0.3), 2: (0.3, 0), 3: (0, -0.3)}


def moving_average(x, window=50):
    x = np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_results(env, q_table, policy, reward_history):
    desc = env.unwrapped.desc.astype(str)  # 4x4 grid of S/F/H/G
    n = desc.shape[0]
    colors = {"S": "#cfe8ff", "F": "#ffffff", "H": "#555555", "G": "#ffe27a"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # --- Left: learning curve ---
    ax1.plot(moving_average(reward_history, 50))
    ax1.set_title("Learning curve (50-episode moving average)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Mean reward")
    ax1.grid(alpha=0.3)

    # --- Right: board with greedy policy arrows + max Q per state ---
    for r in range(n):
        for c in range(n):
            s = r * n + c
            tile = desc[r, c]
            ax2.add_patch(Rectangle((c, r), 1, 1, facecolor=colors[tile],
                                    edgecolor="black"))
            ax2.text(c + 0.08, r + 0.2, f"{tile}", fontsize=9, color="gray")
            if tile in "HG":
                reward = 1 if tile == "G" else 0
                ax2.text(c + 0.5, r + 0.55,
                         f"terminal\nreward={reward}\nV={q_table[s].max():.2f}",
                         ha="center", va="center", fontsize=8,
                         color="white" if tile == "H" else "black")
            if tile in "SF" and q_table[s].max() > 0:
                dx, dy = ARROWS[int(policy[s])]
                ax2.annotate("", xy=(c + 0.5 + dx, r + 0.5 + dy),
                             xytext=(c + 0.5 - dx, r + 0.5 - dy),
                             arrowprops=dict(arrowstyle="->", lw=2, color="crimson"))
                ax2.text(c + 0.5, r + 0.9, f"{q_table[s].max():.2f}",
                         ha="center", fontsize=8)
    ax2.set_xlim(0, n)
    ax2.set_ylim(n, 0)
    ax2.set_aspect("equal")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Greedy policy (arrows) and max Q-value")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)

    # 1) Train without rendering (fast)
    train_env = gym.make("FrozenLake-v1", is_slippery=False)
    train_env.action_space.seed(0)
    q_table, reward_history = train_q_learning(
        train_env, num_episodes=2000, alpha=0.1, gamma=0.99,
        epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995,
        seed=0, max_steps=200,
    )
    policy = extract_greedy_policy(q_table)

    # Terminal (console) output
    print(f"Q-table shape: {q_table.shape}")
    print(f"Mean reward first 100 episodes: {np.mean(reward_history[:100]):.3f}")
    print(f"Mean reward last 100 episodes:  {np.mean(reward_history[-100:]):.3f}")
    print(f"Greedy action at state 0: {greedy_action(q_table, 0)}")
    print(f"Max Q-value at state 0: {max_q_value(q_table, 0):.4f}")
    print(f"Greedy success rate over 100 episodes: "
          f"{evaluate_success_rate(train_env, policy, num_episodes=100, seed=0, max_steps=200):.2f}")

    # State-value grid (V(s) = max_a Q(s,a)); holes (H) and goal (G) are terminal
    desc = train_env.unwrapped.desc.astype(str)
    n = desc.shape[0]
    print("\nState values V(s) = max Q(s,a)   [H = hole, G = goal, terminal]")
    for r in range(n):
        print("  ".join(f"{desc[r, c]}:{q_table[r * n + c].max():.3f}" for c in range(n)))
    print()
    train_env.close()

    # 2) Plots: learning curve + policy arrows (close the window to continue)
    plot_results(train_env, q_table, policy, reward_history)

    # 3) Watch the agent play in a live game window
    play_env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
    for ep in range(3):
        reached = run_greedy_episode(play_env, policy, seed=ep, max_steps=200)
        print(f"Replay {ep + 1}: reached goal = {bool(reached)}")
    play_env.close()
