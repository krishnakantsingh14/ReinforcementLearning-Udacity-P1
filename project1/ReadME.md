# Unity Banana Navigation


## Project details

Project created as 1st project on Udacity Deep Reinforcement Learning nanodegree. The goal of the agent is to gather `yellow` bananas while avoiding the `blue` ones. Here are Unity details of the environment:

```
Unity brain name: BananaBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 37
        Number of stacked Vector Observation: 1
        Vector Action space type: discrete
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
```

That means we work with state vector containing 37 continous values and 4 discrete actions representing moves (forward, backward, turn left, turn right). The environment is considered solved when agents reaches average score of 13.0 on 100 consecutive episodes.

## Getting started


to install python dependencies. 
Then you should be able to run `jupyter notebook` and view `Navigation.ipynb`.
File `ddqn_agent.py` contains agents and neural network class that has been used to approximate Q values using dueling DQN algorithm.

## Instructions

Run `Navigation.ipynb` for further details.


## Solution

Agent is able to average score of 13 in 487 episodes. 
