# CHANGES: Elevator Dispatcher (Group Butternut)

This file lists every change made to the project code, why it was made, and what the results look like afterwards.

**Files changed:** `ElevatorDispatchJesus.py`, `q_agent_elevator.py`, `experiments.py`

---

## 1. `ElevatorDispatchJesus.py` (environment)

| # | Change | Why |
|---|---|---|
| 1 | Added `self.dest_rng = random.Random(seed + 1)`. A rider's destination is now drawn with `dest_rng.choice(...)` instead of `rng.choice(...)`. | Before, one RNG drove both arrivals and destinations. Destinations are only drawn on pickups, so the random stream depended on the agent's actions, and two policies on the same seed saw different passengers. Now `rng` is used for arrivals only, so every policy faces identical arrivals on a given seed. This makes the "trained agent vs RandomAgent on the same seeds" comparison fair. |
| 2 | Added a check right after the `done` check: `if action not in self.actions(self.state): raise ValueError(...)`. | Illegal or misspelled actions (`"up"` on the top floor, `"Open"`) used to be silently ignored. Now they fail loudly, like the `Vaccum` example in the class code. |
| 3 | Docstring: the `rng` entry now says "arrivals only", and a `dest_rng` entry was added. | Keeps the documentation accurate for the code defense. |

Not changed, but worth knowing for the defense:
- Arrival logic is correct: one attempt per step with probability `arrival_prob`, a uniform random floor, and a person is added only if that floor is empty. The effective arrival rate is lower than `arrival_prob` because attempts on occupied floors are discarded, and the gap grows at 0.30.
- The reward is computed from the state before the action, so a rider dropped off this step still costs -1 and earns +5, and a person arriving this step costs nothing until the next step.
- The project's "about 2,500 states" is really state-action pairs. There are 960 states (5 floors x 32 waiting patterns x 6 rider values), and 13 x 192 = 2,496 state-action pairs.

---

## 2. `q_agent_elevator.py` (Q-agent)

The class now defines only `__init__`, `act` and `update`, which are the methods the `Agent` interface requires.

| # | Change | Why |
|---|---|---|
| 1 | Epsilon decay is now implemented, at the end of `update` when `done` is True: `epsilon = max(min_epsilon, epsilon * epsilon_decay)`. | The original stored `epsilon_decay` and `min_epsilon` but never used them, so epsilon stayed at 1.0 and the agent acted randomly forever. It lives in `update` because the agent may only have `act` and `update`, and `done` is its only signal that an episode ended. |
| 2 | Default `epsilon_decay` changed from 0.99 to 0.9998. | At 0.99, epsilon hits the 0.01 floor after about 458 episodes, far too early for a 30,000-episode run. At 0.9998 it takes about 23,000 episodes. |
| 3 | Added a `num_floors=5` constructor argument, and `max_floor = num_floors - 1` (was hard-coded `4`). | Removes a magic number that has to match the environment. |
| 4 | Simplified the legal-action code in `act` and `update`: removed the redundant `and "down" in legal_actions` checks, and `update` takes `max(...)` over a generator instead of building a list with an empty-list guard. | Identical behaviour with less code. The legal-action set is never empty. |
| 5 | Added a full class docstring, an `__init__` docstring and method docstrings. | Matches the environment's documentation style, for the defense. |

Notes:
- The agent works out legal actions from the floor in the state, because it cannot call `env.actions()`. It never reads the environment's private fields.
- The class default `gamma` is still **0.99**, but `experiments.py` passes **0.95**. The class default is not the value actually used. Either change the default to 0.95 or be ready to explain the difference.

---

## 3. `experiments.py`

| # | Change | Why |
|---|---|---|
| 1 | Removed the local `run_episode` and added `from RL_interfaces_v2 import run_episode`. | One shared definition, and it is the class code. |
| 2 | `test()` now uses `run_episode` in a list comprehension. | Shorter code. The scores were verified identical to the previous version. |
| 3 | `train_q` freezes the agent at the end: `epsilon = 0`, `min_epsilon = 0`, `alpha = 0`. | `run_episode` calls `agent.update` every step. `alpha = 0` stops the Q-table changing. `min_epsilon = 0` stops the decay in `update` from pushing epsilon back up to 0.01 after the first test episode. |
| 4 | Removed the manual epsilon-decay line from the training loop. | The agent now decays itself, so keeping it would have decayed twice per episode. |
| 5 | `train_q` has a new `gamma=0.95` parameter, passed to `QAgent`. | See the gamma sweep below. |
| 6 | `SHORT_EPISODES` 500 -> 5000, `LONG_EPISODES` 3000 -> 30000. | The file had the quick-test values. The comments said the real run uses 5,000 and 30,000. |

**Warning:** testing an agent that has not been frozen will keep training it. The freezing lives in `train_q`, so any new test code needs to freeze the agent the same way.

---

## Gamma sweep (arrival_prob = 0.15, 30,000 training episodes, 300 test episodes)

The nearest-call rule scored -22.5.

| Gamma | Training seed 1 | Training seed 2 |
|---|---|---|
| 0.90 | -27.7 | -24.0 |
| 0.95 | -24.4 | -23.1 |
| 0.99 | -33.8 | -29.5 |

**Why 0.99 did worse:** the frozen greedy agent can get stuck in rare states, repeating `open` (which changes nothing) while the penalty grows. With gamma 0.99, 44 of 300 test episodes scored below -60, while the other 256 averaged -18.1. In training, the online updates and the 1% exploration break these loops, so the training curve looked better than the test score.

---

## Final results (average reward, 300 test episodes, test seed 999)

| Policy | 0.05 | 0.15 | 0.30 | Project benchmark (0.15) |
|---|---|---|---|---|
| Q-learning, 30,000 episodes | -5.7 | -24.4 | -64.6 | -20.8 |
| Nearest-call rule | -4.3 | -22.5 | -64.4 | -22.0 |
| Q-learning, 5,000 episodes | -6.0 | -30.9 | -99.9 | -63.3 |
| Random | -62.3 | -153.3 | -226.6 | -160.7 |

- The Q-agent roughly matches the nearest-call rule, with differences of 1 to 2 points at 0.05 and 0.15 and a tie at 0.30. It is not clearly better, and these numbers come from one training seed, so say "roughly matches".
- The learning curve at 0.15 climbs from about -116 to about -21 over training.
- Full output, including the animations, is in `experiment_results.txt`.

## Known issue

The seed-7 animation at arrival_prob 0.15 shows the agent repeating `open` for 48 of 60 steps with four people waiting. This is the stuck-state weakness described above, in a rare, under-trained state. We should decide whether to show a different seed (and say so on the slide) or keep it and explain the limitation.