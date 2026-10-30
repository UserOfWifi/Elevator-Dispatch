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
    def step(self, action) -> tuple: # what is the data type of action?
        """Apply an action. Return (next_state, reward, done)."""

    @abstractmethod
    def actions(self, state) -> list: # what is the data type of state? 
        """Return the list of actions available in `state`."""

# ---------------------------------------------------------------------------
# 1. The Agent interface
# ---------------------------------------------------------------------------

class Agent(ABC):
    """A decision maker. It only ever sees states, actions, and rewards."""

    @abstractmethod
    def act(self, state):
        """Return an action based on `state`."""

    @abstractmethod
    def update(self, s, a, r, s2, done):
        """Update the policy based on state."""


# def run_episode(env, agent):
#     """One episode of the interaction loop from the slide."""
#     s = env.reset()
#     done = False
#     total_reward = 0.0
#     while not done:
#         a = agent.act(s)
#         s2, r, done = env.step(a)          # the only contact with the world
#         agent.update(s, a, r, s2, done)
#         s = s2
#         total_reward += r
#     return total_reward


# def train(env, agent, episodes):
#     """Run `episodes` episodes and return the list of total episode rewards."""
#     return [run_episode(env, agent) for _ in range(episodes)]


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
    """
    Environment for the Bandit RL problem. 

    The RL problem consisting of n slot machines 
    where n=|probs|, a parameter in the constructor.

    Attributes:
    --------------
    probs (list)            - list of floats in [0,1] representing 
                              reward probabilities
    pulls_per_episode (int) - # of pulls per episode
    pulls (int)             - remaining pulls in episode 
    rng (obj)               - random number generator
    state (int)             - not meaningful here; always 0
    done (bool)             - flag when episode is over 
    """

    def __init__(self, probs, pulls=40, seed=0):
        self.probs = list(probs)
        self.pulls_per_episode = pulls
        self.pulls = pulls                # must be positive
        self.rng = random.Random(seed)
        self.state = 0                    # the one and only state
        self.done = False

    def reset(self):
        self.pulls = self.pulls_per_episode
        self.done = False
        return self.state

    def step(self, action):
        reward = 1.0 if self.rng.random() < self.probs[action] else 0.0
        self.pulls -= 1
        if self.pulls == 0:
            self.done = True
        return self.state, reward, self.done   # done is always True

    def actions(self, state):
        return list(range(len(self.probs)))

class Vacuum(Env):
    """
    Two room vacuum environment.

    The vacuum lives in roomms A and B. Each room is either Clean or Dirty, and the vacuum has a  battery that is High or low

    location (str)       - current room(A or B)
    room A (str)         - condition of room A (Clean or Dirty)
    room B (str)         - condition of room B (Clean or Dirty)
    battery (str)        - battery level (High or Low)
    rng (obj)            - random number generator
    done (bool)          - flag when episode is over
    """

    
    def __init__(self, start = ("A", "Dirty", "Dirty", "High"), seed =0):
        self.rng = random.Random(seed)

        self.start = tuple(start)

        self.clean_prob = 0.9
        self.location = self.rng.choice(["A", "B"])
        self.room_A = "Dirty"
        self.room_B = "Dirty"
        self.battery = "High"

# this was more of random thoughs on how i can tackle the question 
    def reset(self):
        pass

    def step(actio):
        pass

    def actions(self, state):
        pass