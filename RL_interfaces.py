from abc import ABC, abstractmethod
import random


# ---------------------------------------------------------------------------
# 1. The Environment interface
# ---------------------------------------------------------------------------

class Env(ABC):
    """The world. Agents must not look inside it -- only call these methods."""

    @abstractmethod
    def reset(self):
        """Start a new episode. Return the initial state."""

    @abstractmethod
    def step(self, action):
        """Apply an action. Return (next_state, reward, done)."""

    @abstractmethod
    def actions(self, state):
        """Return the list of actions available in `state`."""

# ---------------------------------------------------------------------------
# 1. The Agent interface
# ---------------------------------------------------------------------------

class Agent(ABC):
    """A decision maker. It only ever sees states, actions, and rewards."""

    @abstractmethod
    def act(self, state):
        """Choose an action in `state`."""

    @abstractmethod
    def update(self, s, a, r, s2, done):
        """Learn from one observed transition (s, a, r, s2, done)."""


def run_episode(env, agent):
    """One episode of the interaction loop from the slide."""
    s = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        a = agent.act(s)
        s2, r, done = env.step(a)          # the only contact with the world
        agent.update(s, a, r, s2, done)
        s = s2
        total_reward += r
    return total_reward


def train(env, agent, episodes):
    """Run `episodes` episodes and return the list of total episode rewards."""
    return [run_episode(env, agent) for _ in range(episodes)]


# ---------------------------------------------------------------------------
# 2 Example Agents
# ---------------------------------------------------------------------------

class RandomAgent(Agent):
    """Baseline: uniform random actions, never learns (update does nothing)."""

    def __init__(self, actions, seed=0):
        self.actions = list(actions)
        self.rng = random.Random(seed)

    def act(self, state):
        return self.rng.choice(self.actions)

    def update(self, s, a, r, s2, done):
        pass                              # nothing to learn here

class EpsilonGreedyBandit(Agent):
    """Sample-average action values with epsilon-greedy exploration."""

    def __init__(self, n_actions, epsilon=0.1, seed=0):
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q = [0.0] * n_actions        # estimated value of each action
        self.n = [0] * n_actions         # times each action was taken

    def act(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(len(self.q))
        return max(range(len(self.q)), key=lambda a: self.q[a])

    def update(self, s, a, r, s2, done):
        self.n[a] += 1
        self.q[a] += (r - self.q[a]) / self.n[a]   # incremental mean

# ---------------------------------------------------------------------------
# Example Environment: Bandit
# ---------------------------------------------------------------------------

class Bandit(Env):

    def __init__(self, probs, seed=0, pulls_per_episode=40):
        self.probs = list(probs)
        self.rng = random.Random(seed)
        self.state = 0                    # the one and only state
        self.pulls_per_episode = pulls_per_episode # max pulls allowed in 1 episode
        self.pull_count = 0 #how many pulls have happened so far


    def reset(self):
        self.pull_count = 0
        return self.state

    def step(self, action):
        reward = 1.0 if self.rng.random() < self.probs[action] else 0.0
        self.pull_count += 1
        done = self.pull_count >= self.pulls_per_episode
        return self.state, reward, done  

    def actions(self, state):
        return list(range(len(self.probs)))

