# 1. Train a Q-learning agent so it learns to run the elevator
# 2. Test it against two "opponents": the nearest-call rule and random
# 3. Repeat for 3 busy-ness levels (arrival_prob = 0.05, 0.15, 0.30)
# 4. Show the results as a scoreboard and a learning curve
# 5. Animate one episode so we can SEE what the agent learned


from collections import Counter 

from ElevatorDispatchJesus import Elevator
from baseline_agents import NearestCallAgent, ElevatorRandomAgent
from q_agent_elevator import QAgent

TEST_EPISODES = 300 # number of practice games to score each agent
TEST_SEED = 999 # same seed for every agent, so test is fair
SHORT_EPISODES = 500
LONG_EPISODES = 3000


## Run Episode
def run_episode(env, agent):
    """Play one episode. The agent learns after every step."""
    state = env.reset()
    done = False
    total = 0 #score
    while not done:
        action = agent.act(state) # agent picks up or down or open
        next_state, reward, done = env.step(action) # world reacts
        agent.update(state, action, reward, next_state, done) # agent learns
        state = next_state
        total += reward # add this reward to score
    return total 

## Training
def train_q(arrival_prob, episodes, seed=1):
    env = Elevator(seed=seed, arrival_prob=arrival_prob) 
    agent = QAgent(seed=seed, epsilon_decay=0.9998) # epsilon_decay=0.9998 means "explore slowly".  default was (0.99) but
    # agent stopped trying new things after ~460 games and learned worse.
    rewards = []
    for _ in range(episodes):
        total = run_episode(env, agent)
        rewards.append(total)
    # remember the score
        # Epsilon = chance of a random move. gets  a little smaller after each game
        # so the agent explores a lot early and mostly trusts itself later.
        agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
    agent.epsilon = 0.0 # for testing: no random moves, use only what it learned
    return agent, rewards

## Testing
def test(agent, arrival_prob):
    env = Elevator(seed=TEST_SEED, arrival_prob=arrival_prob)
    # same seed = fair test
    totals = []
    for _ in range(TEST_EPISODES):
        state = env.reset()
        done = False
        total = 0
        while not done:
            action = agent.act(state)
            state, reward, done = env.step(action)
            total += reward
        totals.append(total)
    return sum(totals) / len(totals)



## Learning
def learning_curve(rewards, chunk=3000):
    """Print the average training reward for each block of episodes."""
    for start in range(0, len(rewards), chunk):
        block = rewards[start:start + chunk]
        average = sum(block) / len(block)
        print(f"episodes {start + 1:6d} to {start + len(block):6d}: avg reward {average:8.1f}")

## Animation

def animate(agent, arrival_prob):
    """Print one episode, one line per step. E = elevator, * = person waiting."""
    env = Elevator(seed=7, arrival_prob=arrival_prob)
    state = env.reset()
    done = False
    step = 0
    actions_used = []
    while not done:
        action = agent.act(state)
        state, reward, done = env.step(action)
        step += 1
        actions_used.append(action)
        floor, waiting, rider_dest = state
        picture = ""
        for f in range(5):
            car = "E" if f == floor else "."
            person = "*" if waiting[f] == 1 else " "
            picture += "[" + car + person + "]"
        print(f"step {step:2d} {picture} action={action:<5} rider_dest={rider_dest:2d} reward={reward}")
    print("Actions used:", Counter(actions_used))

def show_row(name, score):
    print(f"{name:<32} {score:8.1f}")


## MAIN BLOCK
if __name__ == "__main__":
    trained = {} # trained agent for each arrival_prob
    curves = {} # training scores for each arrival_prob

    for prob in [0.05, 0.15, 0.30]:
        print(f"\n===== arrival_prob = {prob} =====")
        q_short, _ = train_q(prob, SHORT_EPISODES) #barely trained agent
        q_long, rewards = train_q(prob, LONG_EPISODES) #well trained agent
        trained[prob] = q_long
        curves[prob] = rewards

        show_row("Q-learning, long training", test(q_long, prob))
        show_row("Nearest-call rule", test(NearestCallAgent(), prob))
        show_row("Q-learning, short training", test(q_short, prob))
        show_row("Random", test(ElevatorRandomAgent(seed=TEST_SEED), prob))

    print("\n===== learning curve (arrival_prob = 0.15) =====")
    learning_curve(curves[0.15])

    for prob in [0.05, 0.15, 0.30]:
        print(f"\n===== animation, arrival_prob = {prob} =====")
        animate(trained[prob], prob)
        