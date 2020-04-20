## Report 


We used MADDPG algorithm with share critic to solve this environment. 

In this problem, we trained NNs for approximating policy for an individual agent.
 Therefore, we trained two actor NN for policy approximation. On the otherhand, 
 Q values has been approximated using a shared critic NN. Both NNs have identical
  architectures (except the output layer) with two hidden layers (256, 256) and
 relu activation units. The output layer of actor NN has 2 neurons with tanh as
  an activation function. It outputs a 2 dimensional vector and all values lies between -1 and 1. On the other hand, the last layer of critic NN has one neuron. Apart from that, replay buffer and
target network also used. 
 
 
 #####  Hyperparameters
 --------------------
 
 ###### NN training

  - Optimizer: Adam
  - Actor Learning Rate: 1e-4
  - Critic Learning Rate: 2e-4
  - Loss: Mean square loss
  
 ###### MADDPG parameters
  - Learing rate: 1e-4
  - ReplayBuffer: 1e6
  - epsilon: 1.0 
  - Ending epsilon: 0.01 
  - epsilon decay rate: 0.999
  

#### Solution
Environment is solved in 854 episodes. 
Agents are able to achieve 0.50 an average score over 100 episodes. 
[image1]: 
#### Future work 

In future, I would like to explore hyperparameters and use TRPO, PPO algorithms in multi-agent environment.  
