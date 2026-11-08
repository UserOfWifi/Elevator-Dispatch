# UPDATED: Instead of redefining the Env we import from the class code

import random
from RL_interfaces_v2 import Env

# ---------------------------------------------------------------------------
# ELEVATOR ENVIRONMENT
# ---------------------------------------------------------------------------

class Elevator(Env):
    """
        Environment for the Elevator problem. 

        5 floor buildings (floors 0-4), start at 0, with one elevator

        Actions (str):
        Move up, move down, open doors
            - Opening a door lets person out, takes people in (if someone is waiting)

        Reward Function:
            -1 for every person waiting or riding in the state the agent just acted on
            +5 for successful drop-off
        Episode has 60 steps (actions)

        # Added by Princess

        State is a tuple of (current_floor, waiting, rider_dest)
    
        Attributes:
        --------------
        initial_start(tuple)           - Only purpose is to make the reset function look cleaner and also not have 
                                       the reset function hard coded in case start is customized for whatever reason
        max_steps (int)                - Steps per episode, default is 60
        rng (obj)                      - Seeded generator used ONLY for arrivals (whether and where a person appears)
        dest_rng (obj)                 - Seeded generator used ONLY to pick a rider's destination at pickup. Kept separate so
                                       arrivals are identical for every policy on the same seed
        num_floors (int)               - Number of floors
        arrival_prob (float)           - chance a new person appears each step 
        current_floor                  - floor the elevator is on 
        waiting(list)                  - 0/1 per floor (a list inside, a tuple in the state)
        rider_dest(int)                - Initializes at -1 because no one is "in the elevator", once someone enters it gets written a proper value

        steps_left(int)                - Assigns itself with the max_steps, intended to decrease after every step
        done(bool)                     - Once steps_left is 0, this value will go from False to True and end the episode.
        state(tuple)                   - The current state of which floor the elevator is one, people waiting on floors, and floor destination
        """
    
    # I added the arrival_prob parameter to the constructor, 
    # reason been since it was hard coded might be hard for us to test 
    # the other two probabilities for the experiment 
    def __init__(self, start=(0, (0, 0, 0, 0, 0), -1), max_steps=60, seed=42, arrival_prob=0.15): # This seed will allow us to test the same "rng" chances, yall should be able to change it
        """
        Start is a tuple
            First value is starting floor (always starts at 0)
            Second value is 'waiting', essentially tells you if there is a person waiting on a certain floor or not (0 means no one, 1 means someone waiting)
            Third is -1 because it means there is no rider_dest (rider destination) because no one is in the elevator when started
        """
        self.initial_start = start 
        self.max_steps = max_steps  
        self.arrival_prob = arrival_prob 
        self.rng = random.Random(seed)            # arrivals only
        self.dest_rng = random.Random(seed + 1)   # destinations only (so arrivals match across policies)
        
        self.num_floors = 5  # Floors 0 up to 4
        
        # Creates appropriate variables from the start variables
        self.current_floor = start[0]
        self.waiting = list(start[1])
        
        self.rider_dest = start[2]
        
        # Tracking variables
        self.steps_left = self.max_steps
        self.done = False
        self.state = (self.current_floor, tuple(self.waiting), self.rider_dest) 

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
        # the RUntimeError() was used without raise, so it would not do anything 
        # so I updated using raise as it prevents the env from preocessing addiitional 
        # actiosn after the 60 step epsiode has already finished

        if self.done: 
            raise RuntimeError("Episode has finished! ---")

        # Reject illegal or misspelled actions instead of silently ignoring them
        if action not in self.actions(self.state):
            raise ValueError(f"{action} is not allowed in {self.state}")
        
        # Penalty for all people waiting, penalty for holding onto a rider

        waiting_penalty = sum(self.waiting)
        riding_penalty = 1 if self.rider_dest != -1 else 0
        reward = -(waiting_penalty + riding_penalty)

        if action == "down" and self.current_floor > 0: 
            self.current_floor -= 1  

        elif action == "up" and self.current_floor < self.num_floors - 1: 
            self.current_floor += 1

        elif action == "open": 

            # I seperated the drop off snd pickup because the if/elif you had will only have one thing happen per open but as per the guideline
            # opening drops off the rider first, Then picks whoever waits 

            if self.rider_dest == self.current_floor:
                self.rider_dest = -1
                reward += 5.0                   # +5 per drop-off
 
            if self.rider_dest == -1 and self.waiting[self.current_floor] == 1:
                self.waiting[self.current_floor] = 0
                # The new rider wants a random floor other than this one.
                possible_dests = [f for f in range(self.num_floors)
                                  if f != self.current_floor]
                self.rider_dest = self.dest_rng.choice(possible_dests)

        # 2. Stochastic Passenger Generation (Optional background activity) | This is an AI generated comment but I want Ari to take a look at this and explain what is happening here

         #UPDATED: Passenger generation now uses the configurable arrival_prob.
         #At every time step, the environment makes one random arrival attempt.
         # If the probability check succeeds, a random floor is selected.

        # The selected floor can include the elevator's current floor.
        # However, a new passenger is only added if nobody is already
        # waiting on that floor, maintaining one waiting passenger per floor.
        
        if self.rng.random() < self.arrival_prob:
            arrival_floor = self.rng.randrange(0, self.num_floors)
            if self.waiting[arrival_floor] == 0:
                self.waiting[arrival_floor] = 1

            

        self.steps_left -= 1 #Since our counter starts from 60 we are counting down after each successful step
        if self.steps_left <= 0: # Once the counter reaches 0, we assign done to True and go onto the next episode
            self.done = True

        # Pack and return new state tuple
        self.state = (self.current_floor, tuple(self.waiting), self.rider_dest)
        return self.state, reward, self.done

    
    # So now action reads the floor from the state it is given, (it used to self.current and ignored state)
    def actions(self, state):
        floor = state[0]
        valid_actions = ["open"]        # opening the doors is always allowed
        if floor > 0:
            valid_actions.append("down")
        if floor < self.num_floors - 1:
            valid_actions.append("up")
        return valid_actions