
#### Introduction

State space is 33 dimensional vector and each action is a 4 dimensional vector having values
 between -1 and 1.
  The challenge was to obtain an average of 30 scores over 100 episodes.


#### Methods


We used Deep deterministic Policy Gradient (DDPG) algorithm to train an
 agent for solving continuous control problem. 

Two neural networks have been trained to approximate a policy (actor) and Q-values (critic). Both NNs have identical architectures (except the output layer) with two hidden layers (256, 128) and
 relu activation units.
The output layer of actor NN has 4 neurons with tanh as an activation function. It outputs a 4 dimensional vector and all values lies between -1 and 1. On the other hand, the last layer of critic NN has one neuron. Apart from that, replay buffer and
target network also used. 
 
 
 #####  Hyperparameters
 --------------------
 
 ###### NN training

  - Optimizer: Adam
  - Learning Rate: 1e-4
  - Loss: Mean square loss
  
 ###### DDPG parameters
  - Learing rate: 1e-4
  - ReplayBuffer: 1e6
  - epsilon: 1.0 
  - Ending epsilon: 0.01 
  - epsilon decay rate: 0.999
  
 
 #### Results
  
  Results can be seen in Continuous_Control jupyter notebook. The agent is able to achieve thirty average score in consecutive 100 episodes after training for 938 episodes (~950). 
  Weights for actor and critic networks can be found in the repository (solved_actor.pth, solved_critic.pth). 
  
#### Future works
1. PPO, TRPO algorithm
2. Multi-agent environment



 
