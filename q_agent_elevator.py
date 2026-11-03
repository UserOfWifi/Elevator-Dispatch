from abc import ABC, abstractmethod
import random

class Agent(ABC):
    """A decision maker. It only ever sees states, actions, and rewards."""

    @abstractmethod
    def act(self, state):
        """Return an action based on 'state'."""

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """Update the policy based on state."""


class QAgent(Agent):
    """Q-Learning Agent that makes decisions based on elevator's environment"""

    def __init__(self, seed=0, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.01):
        self.actions = ["up", "down", "open"]
        self.q_table = {} # (state, action) where state is tuple & action is str
        self.rng = random.Random(seed)

        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.max_floor = 4

    def act(self, state):
        floor = state[0]

        # Setting boundaries for legal elevator actions
        legal_actions = list(self.actions)
        if floor == 0 and "down" in legal_actions:
            legal_actions.remove("down")
        if floor == self.max_floor and "up" in legal_actions:
            legal_actions.remove("up")

        # Epsilon-Greedy exploration
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)

        # Exploitation
        q_vals = {action: self.q_table.get((state, action), 0.0) for action in legal_actions}
        max_q = max(q_vals.values())

        # Tie-breaker using seed
        best_actions = [action for action, q in q_vals.items() if q == max_q]
        return self.rng.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """Updates the Q-table using a Q-learning update rule."""
        current_q = self.q_table.get((state, action), 0.0)
        
        if done:
            # If the episode ended, there is no next state value
            max_next_q = 0.0
        else:
            # Determine legal actions for the next state to find max Q(next_state, a')
            next_floor = next_state[0]
            next_legal_actions = list(self.actions)
            if next_floor == 0 and "down" in next_legal_actions:
                next_legal_actions.remove("down")
            if next_floor == self.max_floor and "up" in next_legal_actions:
                next_legal_actions.remove("up")
            
            # Find the maximum Q-value for the next state's legal actions
            next_q_values = [self.q_table.get((next_state, a), 0.0) for a in next_legal_actions]
            max_next_q = max(next_q_values) if next_q_values else 0.0

        # Q-learning calculation
        updated_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        # Store the updated value back into the dictionary
        self.q_table[(state, action)] = updated_q