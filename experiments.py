from collections import Counter 

from ElevatorDispatchJesus import Elevator
from baseline_agents import NearestCallAgent, ElevatorRandomAgent
from q_agent_elevator import QAgent
from RL_interfaces_v2 import run_episode   # class code: one episode of the agent/env loop

TEST_EPISODES = 300 
TEST_SEED = 999
SHORT_EPISODES = 5000 
LONG_EPISODES = 30000 


## Training
def train_q(arrival_prob, episodes, seed=1, gamma=0.90):
    env = Elevator(seed=seed, arrival_prob=arrival_prob)
    agent = QAgent(seed=seed, gamma=gamma, epsilon_decay=0.9998)
    rewards = []
    for _ in range(episodes):
        total = run_episode(env, agent)
        rewards.append(total)   
    agent.epsilon = 0.0      
    agent.min_epsilon = 0.0 
    agent.alpha = 0.0        
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
    chunk = max(1, len(rewards) // 10)            
    for start in range(0, len(rewards), chunk):
        block = rewards[start:start + chunk]     
        average = sum(block) / len(block)         
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
        q_short, _ = train_q(prob, SHORT_EPISODES) 
        q_long, rewards = train_q(prob, LONG_EPISODES) 
        trained[prob] = q_long
        curves[prob] = rewards

        show_row("Q-learning, long training", test(q_long, prob))    
        show_row("Nearest-call rule", test(NearestCallAgent(), prob)) 
        show_row("Q-learning, short training", test(q_short, prob))  
        show_row("Random", test(ElevatorRandomAgent(seed=TEST_SEED), prob))  
    print("\n===== learning curve (arrival_prob = 0.15) =====")
    learning_curve(curves[0.15])

    #for prob in [0.05, 0.15, 0.30]:
    print(f"\n===== animation, arrival_prob = 0.15 =====")
    animate(trained[0.15], 0.15)

