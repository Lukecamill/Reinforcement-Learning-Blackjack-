import numpy as np
import random
from collections import defaultdict


class BlackJackAgent:
    # Base class
    def __init__(self, epsilon=0.1, gamma=1.0):
        # Q-table: {state: [value for stand, value for hit]}
        self.q_table = defaultdict(lambda: [0.0, 0.0])  # unseen states initialised to 0,0
        self.epsilon = epsilon  # exploration rate
        self.gamma = gamma      # discount factor

    # Epsilon-greedy policy
    def get_action(self, state):
        player_sum = state[0]
        if player_sum < 12:
            return 1  # always HIT — no risk of bust
        if player_sum == 21:
            return 0  # always STAND — can't do better
        if random.random() < self.epsilon:          # explore
            return random.randint(0, 1)             # stand (0) or hit (1)
        return int(np.argmax(self.q_table[state]))  # exploit: pick highest Q-value

# Monte Carlo

class MonteCarloAgent(BlackJackAgent):

    def __init__(self, **kwargs): # calss the parent class constructor, meaining it calls BlackJackAgent 
        super().__init__(**kwargs)
        # MC tracks all returns to compute a running average
        self.returns_sum   = defaultdict(lambda: [0.0, 0.0])  # total reward per (state, action)
        self.returns_count = defaultdict(lambda: [0, 0])       # visit count per (state, action)
        self.episode_memory = []

    def add_to_memory(self, state, action, reward):
        self.episode_memory.append((state, action, reward))

    # First-visit Monte Carlo update
    def update(self):
        visited_state_actions = set()

        # Gamma = 1 and reward only at terminal step, so G = final reward
        _, _, final_reward = self.episode_memory[-1]
        G = final_reward

        for state, action, _ in self.episode_memory:
            sa_pair = (state, action)

            if sa_pair not in visited_state_actions:  # first-visit check
                self.returns_sum[state][action]   += G # add the episode total reward
                self.returns_count[state][action] += 1 # add one to the count of visits 

                self.q_table[state][action] = (
                    self.returns_sum[state][action] / self.returns_count[state][action]
                )
                visited_state_actions.add(sa_pair)

        self.episode_memory = []  # clear memory for next episode

    # Expose visit counts under the same attribute name as TD agents
    @property
    def visit_counts(self):
        return self.returns_count

# SARSA (on-policy TD)

class SARSAAgent(BlackJackAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # N(s,a): visit counts used to compute adaptive step size alpha = 1/N(s,a)
        self.visit_counts = defaultdict(lambda: [0, 0])

    def update(self, state, action, reward, next_state, next_action, done):
        # Increment visit count BEFORE using it (matches spec: N includes current step)
        self.visit_counts[state][action] += 1
        alpha = 1.0 / self.visit_counts[state][action]

        if done:
            td_target = reward  # no next state
        else:
            td_target = reward + self.gamma * self.q_table[next_state][next_action]

        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += alpha * td_error

# Q-Learning / SARSAMAX (off-policy TD)

class QLearningAgent(BlackJackAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.visit_counts = defaultdict(lambda: [0, 0])

    def update(self, state, action, reward, next_state, done):
        self.visit_counts[state][action] += 1
        alpha = 1.0 / self.visit_counts[state][action]

        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * max(self.q_table[next_state])

        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += alpha * td_error

# Double Q-Learning (off-policy TD)

class DoubleQLearningAgent(BlackJackAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Two independent Q-tables
        self.q1 = defaultdict(lambda: [0.0, 0.0])
        self.q2 = defaultdict(lambda: [0.0, 0.0])
        # Shared visit counts across both tables
        self.visit_counts = defaultdict(lambda: [0, 0])

    # Override: action chosen from the mean of Q1 and Q2
    def get_action(self, state):
        player_sum = state[0]
        if player_sum < 12:
            return 1  # always HIT
        if player_sum == 21:
            return 0  # always STAND
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        mean_q = [
            (self.q1[state][0] + self.q2[state][0]) / 2.0,
            (self.q1[state][1] + self.q2[state][1]) / 2.0,
        ]
        return int(np.argmax(mean_q))

    def update(self, state, action, reward, next_state, done):
        self.visit_counts[state][action] += 1
        alpha = 1.0 / self.visit_counts[state][action]

        if random.random() < 0.5:  # randomly update Q1
            if done:
                td_target = reward
            else:
                best_action = int(np.argmax(self.q1[next_state]))   # argmax from Q1
                td_target   = reward + self.gamma * self.q2[next_state][best_action]  # value from Q2
            self.q1[state][action] += alpha * (td_target - self.q1[state][action])
        else:                      # randomly update Q2
            if done:
                td_target = reward
            else:
                best_action = int(np.argmax(self.q2[next_state]))   # argmax from Q2
                td_target   = reward + self.gamma * self.q1[next_state][best_action]  # value from Q1
            self.q2[state][action] += alpha * (td_target - self.q2[state][action])

        # Keep the public q_table in sync as the mean of Q1 and Q2
        self.q_table[state][action] = (
            self.q1[state][action] + self.q2[state][action]
        ) / 2.0