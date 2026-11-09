# 1. Train a Q-learning agent so it learns to run the elevator
# 2. Test it against two "opponents": the nearest-call rule and random
# 3. Repeat for 3 busy-ness levels (arrival_prob = 0.05, 0.15, 0.30)
# 4. Show the results as a scoreboard and a learning curve
# 5. Animate one episode so we can see what the agent learned


from collections import Counter 

from ElevatorDispatchJesus import Elevator
from baseline_agents import NearestCallAgent, ElevatorRandomAgent
from q_agent_elevator import QAgent
from RL_interfaces_v2 import run_episode   # class code: one episode of the agent/env loop

TEST_EPISODES = 300 # number of practice games to score each agent
TEST_SEED = 999 # same seed for every agent, so test is fair
SHORT_EPISODES = 5000 # barely trained agent
LONG_EPISODES = 30000 # well trained agent


## Training
def train_q(arrival_prob, episodes, seed=1, gamma=0.90):
    env = Elevator(seed=seed, arrival_prob=arrival_prob)
    # epsilon_decay=0.9998 means "explore slowly".  default was (0.99) but
    # agent stopped trying new things after ~460 games and learned worse.
    # gamma=0.95 chosen from a sweep (0.9 / 0.95 / 0.99): 0.99 left the frozen greedy agent
    # stuck in rare states, 0.95 matched the nearest-call rule most closely.
    agent = QAgent(seed=seed, gamma=gamma, epsilon_decay=0.9998)
    rewards = []
    for _ in range(episodes):
        total = run_episode(env, agent)
        rewards.append(total)   # remember the score
    # (epsilon decay happens inside QAgent.update when an episode ends)
    # Freeze the agent for testing. run_episode still calls agent.update, so:
    agent.epsilon = 0.0      # no random moves, use only what it learned
    agent.min_epsilon = 0.0  # update() decays epsilon at episode end; keep it from going back up to 0.01
    agent.alpha = 0.0        # learning rate 0 means update() no longer changes the Q-table
    return agent, rewards

## Testing
def test(agent, arrival_prob):
    """Average score over TEST_EPISODES games on a fixed-seed environment.
    Uses run_episode; learning is switched off beforehand (see train_q), and the
    baseline agents' update() does nothing, so no agent changes during testing."""
    env = Elevator(seed=TEST_SEED, arrival_prob=arrival_prob)  # same seed = fair test
    totals = [run_episode(env, agent) for _ in range(TEST_EPISODES)]
    return sum(totals) / len(totals)


## Learning
def learning_curve(rewards):
    """Show if the agent improves: average score for each block of games."""
    chunk = max(1, len(rewards) // 10)            # split training into 10 blocks
    for start in range(0, len(rewards), chunk):
        block = rewards[start:start + chunk]      # one block of games
        average = sum(block) / len(block)         # average score for the block
        print(f"episodes {start + 1:6d} to {start + len(block):6d}: avg reward {average:8.1f}")

## Animation

def animate(agent, arrival_prob): # new world
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
        floor, waiting, rider_dest = state     # split the state into 3 named pieces
        picture = ""                           # we build the drawing one floor at a time
        for f in range(5):                     # floors 0 to 4
            car = "E" if f == floor else "."   # E if the elevator is on this floor
            person = "*" if waiting[f] == 1 else " "   # * if someone is waiting here
            picture += "[" + car + person + "]"        # stick this floor onto the line
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

        show_row("Q-learning, long training", test(q_long, prob))    # trained a lot
        show_row("Nearest-call rule", test(NearestCallAgent(), prob))  # hand-coded rule
        show_row("Q-learning, short training", test(q_short, prob))  # trained a little
        show_row("Random", test(ElevatorRandomAgent(seed=TEST_SEED), prob))  # no brain
    print("\n===== learning curve (arrival_prob = 0.15) =====")
    learning_curve(curves[0.15])

    #for prob in [0.05, 0.15, 0.30]:
    print(f"\n===== animation, arrival_prob = 0.15 =====")
    animate(trained[0.15], 0.15)

