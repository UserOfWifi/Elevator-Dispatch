from RL_interfaces_v2 import Agent
import random


class QAgent(Agent):
    """
        Tabular Q-learning agent for the Elevator problem.

        The agent only ever sees (state, action, reward, next_state, done).
        It never reads the environment's private fields, and it learns only
        through act() and update(), the two methods required by Agent.

        Attributes:
        --------------
        actions (list)           - All possible action names; legality is filtered per state
        q_table (dict)           - Maps (state, action) -> estimated value. Missing keys count as 0.0
        rng (obj)                - Seeded random generator for exploration and tie-breaking,
                                   so results can be rerun exactly
        alpha (float)            - Learning rate; how far each update moves Q toward its target
        gamma (float)            - Discount factor - how much future reward counts versus immediate reward
        epsilon (float)          - Current chance of taking a random action. Starts high, decays over training
        epsilon_decay (float)    - Multiplier applied to epsilon at the end of each episode (e.g. 0.9998)
        min_epsilon (float)      - Floor for epsilon so the agent never stops exploring entirely
        num_floors (int)         - Number of floors in the building, used to find legal actions
        max_floor (int)          - Index of the top floor (num_floors - 1)
        """

    def __init__(self, seed=0, alpha=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.9998, min_epsilon=0.01, num_floors=5):
        self.actions = ["up", "down", "open"]
        self.q_table = {}  # (state, action)
        self.rng = random.Random(seed)

        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.num_floors = num_floors
        self.max_floor = num_floors - 1

    def act(self, state):
        # Setting boundaries for legal elevator actions
        floor = state[0]
        legal_actions = list(self.actions)
        if floor == 0:
            legal_actions.remove("down")
        if floor == self.max_floor:
            legal_actions.remove("up")

        # Epsilon-Greedy exploration
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)

        # Exploitation
        q_vals = {a: self.q_table.get((state, a), 0.0) for a in legal_actions}
        max_q = max(q_vals.values())

        # Tie-breaker using seed
        best_actions = [a for a, q in q_vals.items() if q == max_q]
        return self.rng.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table.get((state, action), 0.0)

        if done:
            max_next_q = 0.0
        else:
            # Legal actions in the next state
            next_floor = next_state[0]
            next_legal = list(self.actions)
            if next_floor == 0:
                next_legal.remove("down")
            if next_floor == self.max_floor:
                next_legal.remove("up")
            max_next_q = max(self.q_table.get((next_state, a), 0.0)
                             for a in next_legal)

        # Calculation
        target = reward + self.gamma * max_next_q
        self.q_table[(state, action)] = current_q + self.alpha * (target - current_q)

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
