"""
Our learned agent is compared against these (baseline agent whic is nearest call agent)

"""

from RL_interfaces_v2 import Agent, RandomAgent


class ElevatorRandomAgent(RandomAgent):
    """
    Random baseline for the elevator
   Chooses a random legal action (up, down, open) based on the current floor.
    """
    def __init__ (self, seed=0):
        super(). __init__(["up","down", "open"], seed=seed)

    def act(self, state):
        floor, waiting, rider_dest = state
        top = len(waiting) - 1
        legal = ["open"]                  # Opening doors is always allowed
        
        if floor < top:                   # Only allow up if not on the top floor
            legal.append("up")
            
        if floor > 0:                     # Only allow down if not on the ground floor
            legal.append("down")
        
        return self.rng.choice(legal)


class NearestCallAgent(Agent):

    """
    Hand coded baseline: "go to the nearest call"
    It never learns. It only looks at the state (floor, waiting, rider_dest)
    """
    def act(self, state):
        floor, waiting, rider_dest = state

        calls = []                           # Find the floors with waiting passengers
        for f, flag in enumerate(waiting):
            if flag == 1:
                calls.append(f)

        if rider_dest != -1:                 # if someone is inside deliver them first
            target = rider_dest
       
        elif calls:                          # otherwise go to the nearest waiting passenger
            target = min(calls, key=lambda f: abs(floor - f))
        else:
            return "open"

        if target > floor:
            return "up"
        elif target < floor:
            return "down"
        else:
            return "open"

    def update(self, s, a, r, s2, done):  # this baseline follows fixed rules and does not learn 
        pass
    
if __name__ == "__main__":
    agent = NearestCallAgent()
    assert agent.act((3, (1,0,0,0,1), -1)) == "up"
    assert agent.act((2, (0,0,0,0,0), 4)) == "up"
    assert agent.act((0, (0,0,0,0,0), -1)) == "open"
    assert agent.act((2, (1,0,0,0,1), -1)) == "down"  

    rnd = ElevatorRandomAgent(seed=1)
    for _ in range(100):
        assert rnd.act((0, (0,0,0,0,0), -1)) != "down"  # floor 0: never "down"
        assert rnd.act((4, (0,0,0,0,0), -1)) != "up"    # top floor: never "up"

    from collections import Counter
    rnd = ElevatorRandomAgent(seed=2)
    print(Counter(rnd.act((0, (0,0,0,0,0), -1)) for _ in range(1000)))
    print(Counter(rnd.act((2, (0,0,0,0,0), -1)) for _ in range(1000)))

    print("All tests passed")
   