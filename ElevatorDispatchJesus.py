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

class EpsilonGreedyBandit(Agent): # I might be able to use Epsilon Greedy for elevator so it in theory goes for the most effecient route always. 
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
# Example Environment: ELEVATOR
# ---------------------------------------------------------------------------

class Elevator(Env):
    """
        Environment for the Elevator problem. (Based on Bandit)
    
        The RL problem consisting of n slot machines 
        where n=|probs|, a parameter in the constructor.

        5 floor buildings, start at 0

        Action:
        Move up, move down, open doors
            - Opening a door lets person out, takes people in (if someone is waiting)

        Reward Function:
            -1 for each movement while a rider is traveling and for each movement while someone is waiting
            +5 for successful drop-off
        Episode has 60 steps (actions)
    
        Attributes:
        --------------
        initial_start           - Only purpose is to make the reset function look cleaner and also not have 
                                    the reset function hard coded in case start is customized for whatever reason
        max_steps (int)         - Similar idea that I dont want to hard code the max amount of steps
                                    in case step amount passed in is different
        rng (random int)        - Uses the seed to create a random value that helps determine if a random person gets created
        num_floors (int)        - Kept it cause it was easily understandable while reading the code
        
        current_floor           - Takes from the Start parameter, since it's a tuple it takes the first value. First value can be 0 - 4
                                    but in our case it WILL ALWAYS START at 0
        waiting                 - Essentially every floor has a boolean of [0] or [1] if [SOMEONE IS NOT THERE] or [SOMEONE IS THERE] (respecfully)
        rider_dest              - Initializes at -1 because no one is "in the elevator", once someone enters it gets written a proper value

        steps_left              - Assigns itself with the max_steps, intended to decrease after every step
        done                    - Once steps_left is 0, this value will go from False to True and end the episode.
        state                   - The current state of which floor the elevator is one, people waiting on floors, and floor destination
        """
    def __init_(self, start=(0, (0, 0, 0, 0, 0), -1), max_steps=60, seed=42): # This seed will allow us to test the same "rng" chances, yall should be able to change it
        """
        Start is a tuple
            First value is starting floor (always starts at 0)
            Second value is 'waiting', essentially tells you if there is a person waiting on a certain floor or not (0 means no one, 1 means someone waiting)
            Third is -1 because it means there is no rider_dest (rider destination) because no one is in the elevator when started
        """
        self.initial_start = start # Only used to make reset look cleaner and easier to understand
        self.max_steps = max_steps # Same thing, makes it easier to understand rather than spamming 60 everywhere
        self.rng = random.Random(seed) # I have no clue if we need seed so ima just leave it as it is
        
        self.num_floors = 5  # Floors 0 up to 4
        
        # Creates appropriate variables from the start variables
        self.current_floor = start[0]
        self.waiting = list(start[1])
        self.rider_dest = start[2]
        
        # Tracking variables
        self.steps_left = self.max_steps
        self.done = False
        self.state = (self.current_floor, tuple(self.waiting), self.rider_dest) # Same structure as the "start" parameter
        

    def reset(self):
        # Reassigns the starting values from the start 
        self.current_floor = self.initial_start[0]
        self.waiting = list(self.initial_start[1])
        self.rider_dest = self.initial_start[2]
        
        self.steps_left = self.max_steps
        self.done = False
        self.state = (self.current_floor, tuple(self.waiting), self.rider_dest)
        return self.state

    def step(self, action):
        if self.done: # Ends the episode 
            RuntimeError("Episode has finished! ---")
        #Create refined reward system, the bandit one randomly rewards 1 and if it doesnt meet criteria it rewards 0.
        reward = -1

        if action == 0 and self.current_floor > 0: # Move DOWN one floor
            self.current_floor -= 1  

        elif action == 2 and self.current_floor < self.num_floors - 1: # Move UP one floor
            self.current_floor += 1

        elif action == 1: # Drop off and convenient pick up.

            if self.rider_dest == -1 and self.waiting[self.current_floor] == 1: # Determines if no one is inside AND someone is waiting current floor
                self.waiting[self.current_floor] = 0 # Sets person waiting to 0 because zero means no one is waiting (we are about to verify we picked them up)
                # Here we will create a random floor destination for the new person, so the elevator has a goal again. 
                possible_dests = [f for f in range(self.num_floors) if f != self.current_floor] # Generates a list of valid floor numbers (not including current floor)
                self.rider_dest = self.rng.choice(possible_dests) # Sets rider destination to a random value on the list
                # We do not reward pick ups
                
            # Check if we can drop off a rider
            elif self.rider_dest == self.current_floor:
                self.rider_dest = -1
                reward += 5.0  # Reward on drop off

        # 2. Stochastic Passenger Generation (Optional background activity) | This is an AI generated comment but I want Ari to take a look at this and explain what is happening here
        if self.rng.random() < 0.15:  # 15% chance per step a new person calls the elevator

            # Finds all empty floors (Floors with no people waiting and not current floor) and conjures a new person 15% of the time.
            empty_floors = [f for f in range(self.num_floors) if self.waiting[f] == 0 and f != self.current_floor] # List of empty floors
            if empty_floors: #If empty_floors has content... spawn a dude
                spawn_floor = self.rng.choice(empty_floors)
                self.waiting[spawn_floor] = 1

        self.steps_left -= 1 #Since our counter starts from 60 we are counting down after each successful step
        if self.steps_left <= 0: # Once the counter reaches 0, we assign done to True and go onto the next episode
            self.done = True

        # Pack and return new state tuple
        self.state = (self.current_floor, tuple(self.waiting), self.rider_dest)
        return self.state, reward, self.done

    def actions(self, state):
        # Actions: 0 = Down, 1 = Stay, 2 = Up
        valid_actions = [1] # This auto applies 1 to the list, 1 is always valid because staying is always a valid possibility
        
        if self.current_floor > 0: #Checks if Elevator can move DOWN
            valid_actions.append(0)  # Can move down
        if self.current_floor < self.num_floors - 1: #Checks if Elevator can move UP
            valid_actions.append(2)
        return valid_actions # INFORMS THE AGENT ALL VALID MOVEMENTS *******************