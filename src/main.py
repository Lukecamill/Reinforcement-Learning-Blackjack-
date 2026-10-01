import random
import numpy as np
import csv
from environment import Environment
from agent import MonteCarloAgent, SARSAAgent, QLearningAgent, DoubleQLearningAgent

# Epsilon schedules (shared across all algorithms)

EPSILON_CONFIGS = {
    "fixed_0.1":         lambda k: 0.1,
    "decay_linear":      lambda k: 1.0 / k,
    "decay_exp1000":     lambda k: np.exp(-k / 1000.0),
    "decay_exp10000":    lambda k: np.exp(-k / 10000.0),
}

ALGO_NAMES = ["monte_carlo", "sarsa", "qlearning", "double_qlearning"]

TOTAL_EPISODES = 100_000
LOG_INTERVAL   = 1_000

# Episode runners — one per algorithm family

def run_monte_carlo_episode(env, agent, episode_idx):
    state  = env.reset()
    action = agent.get_action(state)
    done   = False

    while not done:
        next_state, reward, done = env.step(action)
        agent.add_to_memory(state, action, reward)
        if not done:
            state  = next_state
            action = agent.get_action(state)

    agent.update()
    return reward


def run_sarsa_episode(env, agent):
    state  = env.reset()
    action = agent.get_action(state)
    done   = False

    while not done:
        next_state, reward, done = env.step(action)
        if not done:
            next_action = agent.get_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done=False)
            state, action = next_state, next_action
        else:
            agent.update(state, action, reward, next_state, None, done=True)

    return reward


def run_qlearning_episode(env, agent):
    state = env.reset()
    done  = False

    while not done:
        action                  = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state

    return reward


def run_double_qlearning_episode(env, agent):
    # Same loop as Q-Learning — agent.update handles double-Q logic internally
    return run_qlearning_episode(env, agent)

# Generic session runner

def run_session(algo_name, epsilon_type, total_episodes=TOTAL_EPISODES):
    env = Environment()

    # Build agent with starting epsilon (will be updated per-episode)
    initial_eps = EPSILON_CONFIGS[epsilon_type](1)
    if algo_name == "monte_carlo":
        agent = MonteCarloAgent(epsilon=initial_eps)
    elif algo_name == "sarsa":
        agent = SARSAAgent(epsilon=initial_eps)
    elif algo_name == "qlearning":
        agent = QLearningAgent(epsilon=initial_eps)
    else:  # double_qlearning
        agent = DoubleQLearningAgent(epsilon=initial_eps)

    label = f"{algo_name}_{epsilon_type}"
    print(f"  Starting: {label}")

    history_log = []                                             # cumulative stats every 1k eps
    cumulative  = {"wins": 0, "losses": 0, "draws": 0}
    last_10k    = {"wins": 0, "losses": 0, "draws": 0}

    for i in range(1, total_episodes + 1):
        # Update epsilon for this episode
        agent.epsilon = EPSILON_CONFIGS[epsilon_type](i)

        # Run one episode
        if algo_name == "monte_carlo":
            reward = run_monte_carlo_episode(env, agent, i)
        elif algo_name == "sarsa":
            reward = run_sarsa_episode(env, agent)
        elif algo_name == "qlearning":
            reward = run_qlearning_episode(env, agent)
        else:
            reward = run_double_qlearning_episode(env, agent)

        # Accumulate stats
        if reward == 1:
            cumulative["wins"]   += 1
        elif reward == -1:
            cumulative["losses"] += 1
        else:
            cumulative["draws"]  += 1

        # Track last 10k for dealer advantage
        if i > total_episodes - 10_000:
            if reward == 1:    last_10k["wins"]   += 1
            elif reward == -1: last_10k["losses"] += 1
            else:              last_10k["draws"]  += 1

        # Log cumulative totals every LOG_INTERVAL episodes
        if i % LOG_INTERVAL == 0:
            history_log.append([
                i,
                cumulative["wins"],
                cumulative["losses"],
                cumulative["draws"],
            ])
            if i % 10_000 == 0:
                print(f"    Episode {i:>6}: "
                      f"W={cumulative['wins']}  "
                      f"L={cumulative['losses']}  "
                      f"D={cumulative['draws']}")

    # Save CSVs
    save_history_csv(f"results_{label}.csv",  history_log)
    save_policy_csv(f"policy_{label}.csv",    agent)
    save_counts_csv(f"counts_{label}.csv",    agent)

    W  = last_10k["wins"]
    L  = last_10k["losses"]
    da = (L - W) / (L + W) if (L + W) > 0 else 0.0
    print(f"    Dealer advantage (last 10k): {da:.4f}\n")

    return agent, da

# CSV helpers

def save_history_csv(filename, data):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "CumWins", "CumLosses", "CumDraws"])
        writer.writerows(data)
    print(f"    Saved {filename}")


def save_policy_csv(filename, agent):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PlayerSum", "DealerCard", "UsableAce",
                         "Q_Stand", "Q_Hit", "BestAction"])
        for state, values in agent.q_table.items():
            player_sum, dealer_card, usable_ace = state
            best = "S" if values[0] >= values[1] else "H"
            writer.writerow([player_sum, dealer_card, usable_ace,
                             values[0], values[1], best])
    print(f"    Saved {filename}")


def save_counts_csv(filename, agent):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PlayerSum", "DealerCard", "UsableAce", "Action", "Count"])
        for state, counts in agent.visit_counts.items():
            player_sum, dealer_card, usable_ace = state
            for action_idx, count in enumerate(counts):
                action_name = "Stand" if action_idx == 0 else "Hit"
                writer.writerow([player_sum, dealer_card, usable_ace, action_name, count])
    print(f"    Saved {filename}")

# Main

if __name__ == "__main__":
    dealer_advantages = {}

    for algo in ALGO_NAMES:
        print(f"\n{'='*60}")
        print(f"  ALGORITHM: {algo.upper()}")
        print(f"{'='*60}")
        for eps_type in EPSILON_CONFIGS:
            agent, da = run_session(algo, eps_type)
            dealer_advantages[f"{algo}_{eps_type}"] = da

    # Summary CSV
    with open("dealer_advantages_all.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "EpsilonType", "DealerAdvantage"])
        for key, da in dealer_advantages.items():
            algo, *eps_parts = key.split("_", 1)
            writer.writerow([key.split("_")[0], "_".join(key.split("_")[1:]), da])
    print("Saved dealer_advantages_all.csv")
    print("\nAll 16 experiments complete.")