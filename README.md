# Two improvements based on MADDPG algorithm


## 1. Introduction

Based on MADDPG algorithm, there are mainly two improvements:

- maddpg_IU-master：the code implementation of updating the agents strategies by iterative update.
- maddpg_IUUR-master：the code implementation of updating the agent strategies by iterative update and unified representation.

The experimental environment is multiagent-particle-envs-master, which is a multi-agent environment that is installed in the same way as MADDPG-env.

## 2. Environment

There are mainly two environments:fully-cooperative and mixed cooperative-competitive:

- fully-cooperative environments(Spread):Agents perceive the environment on its own perspective and cooperate with each other to reach different destinations.
- mixed cooperative-competitive environments(Predator-Prey):Agents are divided into predators and preys. The predators need to cooperate with each other to catch the preys.

 In order to compare the influence of the number of agents on the algorithm, it is designed the control groups by increasing the number of agents. 
 
## 3. Install

Installation method and dependency package versions are the same as MAPPDG:

- To experimental environment install: cd into the root directory and type pip install -e .
- Known dependencies: Python (3.5.4), OpenAI gym (0.10.5), tensorflow (1.8.0), numpy (1.14.5)

## 4. Results

### ①fully-cooperative environment
We set up a simple environment with three agents (Spread_3) and a complex environment with ten agents(Spread_10).
&nbsp;&nbsp; &nbsp; <img src='https://github.com/DreamChaser128/IUUR-for-Multi-Agent-Reinforcement-Learning/blob/master/images/Spread_3.png' alt='Spread_3' width='200' height='200'> &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; <img src='https://github.com/DreamChaser128/IUUR-for-Multi-Agent-Reinforcement-Learning/blob/master/images/Spread_10.png' alt='Spread_10' width='200' height='200'><br />
Spread_3
We run five random seeds for each environment and compare the performance among MADDPG, IU and IUUR.
图。

As can be seen from the figure, IUUR converges quickly and after 20,000 episodes, it has exceeded MADDPG and maintained a steady rise. 

### ②mixed cooperative-competitive environment (the baseline is MADDPG vs MADDPG)
We set up three chase one as simple scenes(Predator_3-prey_1) and six chase two as complex scenes(Predator_6-prey_2).
图。

We run five random seeds for each environment and compare the performance among MADDPG, IU and IUUR.
- **performance comparison in Predator_3-Prey_1**
  - the prey is MADDPG while the predators are replaced by IU and IUUR:
  图。
  - the predator is MADDPG while the preys are replaced by IU and IUUR:
  图。

  IUUR outperforms MADDPG a lot. IU’s performance is slightly worse than that of MADDPG which is out of our expectation.

- **performance comparison in Predator_6-prey_2**
  - the prey is MADDPG while the predators are replaced by IU and IUUR:
   图。
  - the predator is MADDPG while the preys are replaced by IU and IUUR:
   图。

  IU outperforms MADDPG a lot while IUUR’s performance is worse than that of MADDPG. The reason is that as the number of agents increases, nonstationarity arises in multi-agent reinforcement learning gets more serious. 

