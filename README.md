# DRL-ND-Project_2

## Description

In this project I trained a SAC agent to solve the unity environment "Reacher" which consist in a Double-jointed arm which must move to diferent target locations, the states and actions in the environment consists as follows:

* Vector Observation space: 33 variables corresponding to position, rotation, velocity, and angular velocities of the two arm rigid bodies.
* Vector Action space: (Continuous) Size of 4, corresponding to torque applicable to two joints.
* Visual Observations: None

And a reward of +0.1 is provided for each step that the agent's hand is in the goal location, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

![Environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/images/reacher.png?raw=true)

## Getting Started

To train or test the sac_agent.

1. Clone this repo.
2. Create a virtual environment.
2. Over the environment run `pip install -r requirements.txt`

## Instructions

All the code to train or to test the sac agent is contained inside the **'Continuous_Control.ipynb'** file and following the cell instructions you can train an agent from scratch or test it loading the saved model.

## Results

![Reacher testing](./media/Reacher.gif)

The agent was able to achive a reward of +30 after 298 episodes. 

## Citations

Most of the code is based on the github repo "Deep-Reinforcement-Learning-Algorithms/BipedalWalker-Soft-Actor-Critic" from @rafael1s 