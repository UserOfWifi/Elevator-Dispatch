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
    """
    where n=|probs|, a parameter in the constructor.

    Attributes:
    --------------
    probs (list)   - list of floats in [0, 1] representing
                    reward probabilities
    pulls_per_episode (int) - # of pulls per episode
    pulls (int)             - remaining pulls in episode
    rng (obj)               - random number generator
    state (int)             - not meaningful here; always -
    done (bool)             - flag when episode is over
    """

    def __init__(self, probs, seed=0, pulls=40):
        self.probs = list(probs)
        self.rng = random.Random(seed)
        self.state = 0                    # the one and only state
        self.pulls_per_episode = pulls
        self.pulls = pulls
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
        return self.state, reward, self.done  # done is always True

    def actions(self, state):
        return list(range(len(self.probs)))

class Vaccum(Env): # is this the name I want for the class?
    """
    Two room robot vaccum environment
    """

    def __init__(self, start:tuple=('A', 'Dirty', 'Dirty', 'High'), seed=0):
        """Initializing the constructor"""
        # Should these be attributes? Should they be passed in?
        self.p_clean = 0.9     # cleaning succeeds
        self.p_dirty = 0.1     # clean room not being cleaned becomes dirty
        self.p_drop = 0.3      # non-dock action drops the battery
        self.dead_reward = -10.0  # reward design
        self.start = tuple(start)  # Do I need this if I have state???
        self.rng = random.Random(seed)
        self.state = self.start
        self.done = False
    
    def reset(self):
        """Start a new episode. Return the initial state."""
        self.state = self.start
        self.done = False
        return self.state

        
    def actions(self, state):
        """Return the list of actions available in `state`."""
        if state == self.DEAD:
            return [] # why an empty list if dead?
        acts = ['clean', 'Move']
        if state[0] == 'A':
            acts.append('Dock')
        return acts

    def _redirty(self, room_status: str) -> str:    # private/protected

        if room_status == 'Clean' and self.rng.random() < self.p_dirty:
            return 'Dirty'
        return room_status

    def step(self, action):
        if self.done:
            raise RuntimeError("episode is over; call reset() first")
        if action not in self.actions(self.state):
            raise ValueError(f"{action} is not allowed in {self.state}")  

        room, a_stat, b_stat, battery = self.state

        # --- rooms ---
        if action == 'Clean':
            # The robot's room is being cleaned: Clean w.p. p_clean,
            # otherwise it stays exactly as it was.
            if self.rng.random() < self.p_clean:
                if room == 'A':
                    a_stat = 'Clean'
                else:
                    b_stat = 'Clean'    

