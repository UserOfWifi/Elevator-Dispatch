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
        rng (obj)                      - Uses the seed to create a random value that helps determine if a random person gets created
        dest_rng (obj)                 - Another random value based on the seed but it's (seed + 1) to separate arrival_prob rng and destination rng
        num_floors (int)               - Number of floors (hard coded)

        current_floor                  - floor the elevator is on 
        waiting(list)                  - 0/1 per floor (a list inside, a tuple in the state)
        arrival_prob (float)           - % chance a person appears on a floor 
        rider_dest(int)                - Initializes at -1 because no one is "in the elevator", once someone enters it gets written a proper value

        steps_left(int)                - Assigns itself with the max_steps, intended to decrease after every step
        done(bool)                     - Once steps_left is 0, this value will go from False to True and end the episode.
        state(tuple)                   - The current state of which floor the elevator is one, people waiting on floors, and floor destination
        """
    
    # All parameter variables are done so that we can change a value if we want to for testing. 
    def __init__(self, start=(0, (0, 0, 0, 0, 0), -1), max_steps=60, seed=42, arrival_prob=0.15): # This seed will allow us to test the same "rng" chances, yall should be able to change it
        """
        Start is a tuple
            First value is starting floor (always starts at 0)
            Second value is 'waiting', essentially tells you if there is a person waiting on a certain floor or not (0 means no one, 1 means someone waiting)
            Third is -1 because it means there is no rider_dest (rider destination) because no one is in the elevator when started
        """
        self.initial_start = start 
        self.max_steps = max_steps  
        self.rng = random.Random(seed) 
        self.dest_rng = random.Random(seed + 1) # separate rng for destinations, so that the arrival_prob rng is not affected by the destination rng
        
        self.num_floors = 5  # Floors 0 up to 4
        
        # Creates appropriate variables from the start variables
        self.current_floor = start[0]
        self.waiting = list(start[1])
        self.arrival_prob = arrival_prob 
        
        self.rider_dest = start[2]
        
        # Tracking variables
        self.steps_left = self.max_steps
        self.done = False
        self.state = (self.current_floor, tuple(self.waiting), self.rider_dest) 
        self._assign_start_destinations()

    def _new_destination(self, floor): # Used in Princess's addition to the step function to generate a random floor (that isnt used)
        return self.dest_rng.choice([f for f in range(self.num_floors) if f != floor])

    # Initialize hidden destinations for any waiting passengers.
    # -1 means no passenger is waiting at that floor.
    def _assign_start_destinations(self): # If the floors part in the start vairable has people, set it as a destimation.
        self._waiting_dest = [-1] * self.num_floors
        for f in range(self.num_floors): # From floor to top floor check
            if self.waiting[f] == 1:
                self._waiting_dest[f] = self._new_destination(f)

    def reset(self):
        # Reassigns the starting values from the start 
        self.current_floor = self.initial_start[0]
        self.waiting = list(self.initial_start[1])
        self.rider_dest = self.initial_start[2]
        self._assign_start_destinations()
        
        self.steps_left = self.max_steps
        self.done = False
        self.state = (self.current_floor, tuple(self.waiting), self.rider_dest)
        return self.state

    def step(self, action):

        if self.done: 
            raise RuntimeError("Episode has finished! ---") # Princess added "raise" so it would actually stop

        legal = self.actions(self.state) # Princess also added this in case the code messes up and adds an illegal action
        if action not in legal:
            raise ValueError(
                f"Invalid action {action!r} in state  {self.state}; legal actions: {legal}")

        waiting_penalty = sum(self.waiting) # This little block is the penalties 
        riding_penalty = 1 if self.rider_dest != -1 else 0
        reward = -(waiting_penalty + riding_penalty)

        if action == "down": 
            self.current_floor -= 1  

        elif action == "up" : 
            self.current_floor += 1

        elif action == "open": #No "else if" here because both can happen "at once"

            if self.rider_dest == self.current_floor: #Drop dude off
                self.rider_dest = -1
                reward += 5.0
 
            if self.rider_dest == -1 and self.waiting[self.current_floor] == 1: #Pick dude up (sometimes after dropping of a dude)
                self.waiting[self.current_floor] = 0
                #Use the destination assigned when the passenger appeared,
                # rather than choosing another destination at pickup.
                self.rider_dest = self._waiting_dest[self.current_floor]

                #Clear the stored destination because the passenger is now inside.
                self._waiting_dest[self.current_floor] = -1

         #Princess added the "arrival_prob" instead of the hardcoded "0.15" so we could experiment with how many people spawn in. 
        if self.rng.random() < self.arrival_prob:
            arrival_floor = self.rng.randrange(self.num_floors)

            if self.waiting[arrival_floor] == 0:
                self.waiting[arrival_floor] = 1
                self._waiting_dest[arrival_floor] = self._new_destination(arrival_floor)

        self.steps_left -= 1 #Counter starts at 60
        if self.steps_left <= 0: # On to next episode
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
    