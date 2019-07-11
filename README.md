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

- To environment install: cd into the root directory(multiagent-particle-envs-master) and type pip install -e .
- Known dependencies: Python (3.5.4), OpenAI gym (0.10.5), tensorflow (1.8.0), numpy (1.14.5)

## 4. Results

### ①fully-cooperative environment
We set up a simple environment with three agents (Spread_3) and a complex environment with ten agents(Spread_10).
 &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  <img src='https://github.com/DreamChaser128/IUUR-for-Multi-Agent-Reinforcement-Learning/blob/master/images/Spread_3.png' alt='Spread_3' width='270' height='270'> &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; <img src='https://github.com/DreamChaser128/IUUR-for-Multi-Agent-Reinforcement-Learning/blob/master/images/Spread_10.png' alt='Spread_10' width='270' height='270'><br />
 &nbsp;&nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;**Spread_3**&nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;**Spread_10**<br />

We run **five random seeds** for each environment and compare the performance among MADDPG, IU and IUUR.

 &nbsp;&nbsp; <img src='https://github.com/DreamChaser128/IUUR-for-Multi-Agent-Reinforcement-Learning/blob/master/images/Spread_3_comparison.png' alt='Spread_3_comparison' width='300' height='300'> &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<img src='https://github.com/DreamChaser128/IUUR-for-Multi-Agent-Reinforcement-Learning/blob/master/images/Spread_10_comparison.png' alt='Spread_10_comparison' width='300' height='300'><br />
 &nbsp;&nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp;**Spread_3_comparison**&nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;**Spread_10_comparison**<br />

As can be seen from the figure, IUUR converges quickly and after 20,000 episodes, it has exceeded MADDPG and maintained a steady rise. 

### ②mixed cooperative-competitive environment (the baseline is MADDPG vs MADDPG)
We set up three chase one as simple scenes(Predator_3-prey_1) and six chase two as complex scenes(Predator_6-prey_2).
 &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  <img src='https://github.com/DreamChaser128/IUUR-for-Multi-Agent-Reinforcement-Learning/blob/master/images/Predator_3-Prey_1.png' alt='Predator_3-Prey_1' width='270' height='270'> &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; <img src='https://github.com/DreamChaser128/IUUR-for-Multi-Agent-Reinforcement-Learning/blob/master/images/Predator_6-Prey_2.png' alt='Predator_6-Prey_2' width='270' height='270'><br />
 &nbsp;&nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;**Predator_3-Prey_1**&nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;**Predator_6-Prey_2**<br />

We run **five random seeds** for each environment and compare the performance among MADDPG, IU and IUUR.
- **performance comparison in Predator_3-Prey_1**
  - the prey is MADDPG while the predators are replaced by IU and IUUR:
   &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  <img src='https://github.com/DreamChaser128/IUUR-for-Multi-Agent-Reinforcement-Learning/blob/master/images/Predator_3-Prey_1_predator_comparison.png' alt='Predator_3-Prey_1_predator_comparison' width='270' height='270'> 
  - the predator is MADDPG while the preys are replaced by IU and IUUR:
  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  <img src='https://github.com/DreamChaser128/IUUR-for-Multi-Agent-Reinforcement-Learning/blob/master/images/Predator_3-Prey_1_prey_comparison.png' alt='Predator_3-Prey_1_prey_comparison' width='270' height='270'> 

  **IUUR outperforms MADDPG a lot while IU’s performance is slightly worse than that of MADDPG which is out of our expectation.**

- **performance comparison in Predator_6-prey_2**
  - the prey is MADDPG while the predators are replaced by IU and IUUR:
   &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  <img src='https://github.com/DreamChaser128/IUUR-for-Multi-Agent-Reinforcement-Learning/blob/master/images/Predator_6-Prey_2_predator_comparison.png' alt='Predator_6-Prey_2_predator_comparison' width='270' height='270'> <br/>
  - the predator is MADDPG while the preys are replaced by IU and IUUR:
   &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  <img src='https://github.com/DreamChaser128/IUUR-for-Multi-Agent-Reinforcement-Learning/blob/master/images/Predator_6-Prey_2_prey_comparison.png' alt='Predator_6-Prey_2_prey_comparison' width='270' height='270'> 

  **IU outperforms MADDPG a lot while IUUR’s performance is worse than that of MADDPG. The reason is that as the number of agents increases, nonstationarity arises in multi-agent reinforcement learning gets more serious.**
 
## 5. Conclusions
- This paper presents iteration updating and unified representation. lterative update is used to stabilize the environment and unified representation take the advantages of tensor compute to save memory and speed up the interaction with environment. 
- Though our experiments are based on MADDPG, this method is also suitable for most of multi-agent algorithms like IQL, VDN, QMIX etc.

## 6. Future Work
- due to the limited computing resources, we only expand the number of agents to a certain extent, which can be further verified in more complex environments.
- At present, we only simply control the learning frequency of iterative update hyperparameter K through experience, which is a research direction in the future.
- how to realize the iterative update method in this unified representative network,This will be further improved in the future work. (considering the value fixing method based on Bellman Equation can only guarantee a smaller L_2 norm of its gradients)   

