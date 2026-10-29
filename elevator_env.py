from RL_interfaces import Env

class Elevator(Env):
    def reset(self):
        self.floor = 0
        self.waiting = (0, )*5
        self.rider_dest = -1
        self.time_step = 0 # this one is not returned because we are only using it to internally track when an episode should end
        return (self.floor, self.waiting, self.rider_dest) # there probably is an other way to make we return these 3