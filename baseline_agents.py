"""
Our learned agent is compared against these (baseline agent whic is nearest call agent)

"""

from RL_interfaces_v2 import Agent, RandomAgent


class ElevatorRandomAgent(RandomAgent):
    """
    Random baseline for the elevator
    Note: it can choose "up" on the top floor or "down" on the floor 0, so the environment must decide what an illegal move does(e.g nothing happens) (waiting for the environment to be built)
    """
    def __init__ (self, seed=0):
        super(). __init__(["up","down", "open"], seed=seed)


class NearestCallAgent(Agent):

    """
    Hand coded baseline: "go to the nearest call"
    It never learns. It only looks at the state (floor, waiting, rider_dest)
    """
    def act(self, state):
        floor, waiting, rider_dest = state

        calls = []
        for f, flag in enumerate(waiting):
            if flag == 1:
                calls.append(f)

        if rider_dest != -1:
            target = rider_dest
        elif calls:
            target = min(calls, key=lambda f: abs(floor - f))
        else:
            return "open"

        if target > floor:
            return "up"
        elif target < floor:
            return "down"
        else:
            return "open"

    def update(self, s, a, r, s2, done):
        pass
    
if __name__ == "__main__":
    agent = NearestCallAgent()
    assert agent.act((3, (1,0,0,0,1), -1)) == "up"

    print("allllllll")
   