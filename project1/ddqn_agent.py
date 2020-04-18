import numpy as np
import random
from collections import namedtuple, deque

# from dueling_model import * 
# from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim



import torch
from torch import nn
import torch.nn.functional as F



class Dueling_QNetwork(nn.Module):
    """
    Implementation of dueling DQN architecture. See: https://arxiv.org/pdf/1511.06581.pdf
    """
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=128):
        """
        Initialize parameters and build model.
        :param state_size: Dimension of the states
        :param action_size: Dimension of the actions
        :param seed: seed for RNG
        :param fc1_units: First hidden layer node count
        :param fc2_units: Second hidden layer node count
        """
        super(Dueling_QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # value layer
        self.fc_v = nn.Linear(fc2_units, 1)
        # advantage layer
        self.fc_a = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """
        Forward function for the network.
        :param state: input state
        :return: Tensor output of the last hidden layer
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        a = self.fc_a(x)
        # Combine the value and advantage streams to final output.
        # Nomalized a with minus a.mean
        x = v + (a - a.mean(1).unsqueeze(1))
        return x
















BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Double_DQN_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, network_type = None, filename=None):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)


        print("Using double Dueling network")
        self.qnetwork_local = Dueling_QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = Dueling_QNetwork(state_size, action_size, seed).to(device)
            
            
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        if filename:
            weights = torch.load(filename)
            self.qnetwork_local.load_state_dict(weights)
            self.qnetwork_target.load_state_dict(weights)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state) #same as self.qnetwork_local.forward(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## ------------------Double DQN
        #Get the local net next State value
        qns_local = self.qnetwork_local.forward(next_states)
        #Get the argmax a of state of local net 
        _, qnsa_local_argmax_a = torch.max(qns_local, dim=1) 
        #Get the target net next state value
        qns_target = self.qnetwork_target.forward(next_states)
        #Use the argmax a of local net to get the next state Q in target net
        qnsa_target = qns_target[torch.arange(BATCH_SIZE, dtype=torch.long), qnsa_local_argmax_a.reshape(BATCH_SIZE)]

        #Get the TD target
        qnsa_target = qnsa_target * (1 - dones.reshape(BATCH_SIZE)) 
        qnsa_target = qnsa_target.reshape((BATCH_SIZE,1))
        TD_target = rewards + gamma * qnsa_target
        
        #Get the train value
        qs_local = self.qnetwork_local.forward(states)
        qsa_local = qs_local[torch.arange(BATCH_SIZE, dtype=torch.long), actions.reshape(BATCH_SIZE)]
        qsa_local = qsa_local.reshape((BATCH_SIZE,1))
        
        #Backpropagation
        loss = F.mse_loss(qsa_local, TD_target) # mean square loss

        self.optimizer.zero_grad() #clears the gradients
        loss.backward()
        self.optimizer.step()


        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)