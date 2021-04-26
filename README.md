# Door-Key-Robotic-Planning via Dynamic Programming

## Overview
In this project, we implement dynamic programming for the Door-key problem. We are to get the agent (red arrow) to the goal location (green cell) in the fewest number of steps possible.

## Installation

- Install [gym-minigrid](https://github.com/maximecb/gym-minigrid)
- Install dependencies
```bash
pip install -r requirements.txt
```
## Code 
### 2. utils.py
You might find some useful tools in utils.py
- **step()**: Move your agent
- **generate_random_env()**: Generate a random environment for debugging
- **load_env()**: Load the test environments
- **save_env()**: Save the environment for reproducing results
- **plot_env()**: For a quick visualization of your current env, including: agent, key, door, and the goal
- **draw_gif_from_seq()**: Draw and save a gif image from a given action sequence.
### 3. example.py
The example.py shows you how to interact with the utilities in utils.py, and also gives you some examples of interacting with gym-minigrid directly.


## Results
<img src="GIFs/Known Map GIFs/8x8-normal.gif">
<img src="GIFs/Known Map GIFs/8x8-direct.gif">
<img src="GIFs/Known Map GIFs/8x8-shortcut.gif">
<img src="GIFs/Random Map GIFs/random3.gif">
<img src="GIFs/Random Map GIFs/random5.gif">
