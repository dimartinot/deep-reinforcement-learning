import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

import ddpg_agent

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

UPDATE_NUMBER = 10      # Number of updates at every time step
UPDATE_INTERVAL = 20    # Number of timesteps waited between updates

EPSILON = 1.0           # Coefficient of noise added at every timestep: balance the exploration/exploitation dilemna
EPSILON_DECAY = 1e-6    # decay rate for epsilon

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PrioritizedAgent(ddpg_agent.Agent):
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super().__init__(state_size, action_size, random_seed)
        
        self.epsilon = EPSILON
        
        # Replay memory
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    
    def step(self, state, action, reward, next_state, done, timestep=1):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        td_error = self.td_error(state, action, reward, next_state, done).item()
        
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done, td_error)
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and timestep % UPDATE_INTERVAL == 0:
            for i in range(UPDATE_NUMBER):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def td_error(self, state, action, reward, next_state, done, gamma=GAMMA):
        
        state_cp = torch.from_numpy(state).float().to(device) if type(state) is np.ndarray else state.clone()
            
        next_state_cp = torch.from_numpy(next_state).float().to(device) if type(next_state) is np.ndarray else next_state.clone()
        
        action_cp = torch.from_numpy(action).float().to(device) if type(action) is np.ndarray else action.clone()
        
        next_state_cp = next_state_cp.unsqueeze(0)
        if (len(state_cp.shape) != 2):
            state_cp = state_cp.unsqueeze(0)
        if (len(action_cp.shape) != 2):
            action_cp = action_cp.unsqueeze(0)
        
        self.actor_target.eval()
        self.critic_target.eval()
        self.critic_local.eval()
        
        
        with torch.no_grad():
            action_next = self.actor_target(next_state_cp)
            Q_targets_next = self.critic_target(next_state_cp, action_next)
            # Compute Q targets for current states (y_i)
            Q_target = reward + (gamma * Q_targets_next * (1 - done))
            # Compute critic loss
            Q_expected = self.critic_local(state_cp, action_cp)

            td_error = Q_target - Q_expected

            self.actor_target.train()
            self.critic_target.train()
            self.critic_local.train()

            return np.abs(td_error.cpu().numpy())
                
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        if (len(state.shape) != 2):
            state = state.unsqueeze(0)
            
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.epsilon * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done, index) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, indexes = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        
        for i in range(len(states)):
            state, action, reward, next_state, done, index = states[i], actions[i], rewards[i], next_states[i], dones[i], indexes[i]
            
            td_error = Q_targets[i] - Q_expected[i]
            
            if torch.is_tensor(state):
                state = np.array(state.cpu())
            if torch.is_tensor(action):
                action = np.array(action.cpu())
            if torch.is_tensor(reward):
                reward = np.array(reward.cpu())
            if torch.is_tensor(next_state):
                next_state = np.array(next_state.cpu())
            if torch.is_tensor(done):
                done = np.array(done.cpu())
                        
            self.memory.update_td_error(index,state,action,reward,next_state, done, td_error.item())      
        
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)  
        
        # ---------------------------- update noise ---------------------------- #
        self.epsilon -= EPSILON_DECAY
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)   
    
class PrioritizedReplayBuffer(ddpg_agent.ReplayBuffer):
    """Fixed-size buffer to store experience tuples using a priority sampling method"""
    
    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=0.7, beta=0.9, constant=.1):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (int): parameter correcting how non-uniform the sampling is (closer to 0 -> completely uniform, closer to 1 -> completely non-uniform)
            beta (int): parameter describing the importance of the sampling weights in the update rule
            constant (int): parameter used to make non valuable items more likely to be picked
        """
        super().__init__(action_size, buffer_size, batch_size, seed)
        
        # used to determine sampling probability
        self.sampling_td_errors = deque(maxlen=buffer_size) 
        
        self.alpha = alpha
        self.beta = beta
        self.constant = constant
    
    def add(self, state, action, reward, next_state, done, td_error):
        
        if td_error < 0:
            td_error = - td_error
        
        self.sampling_td_errors.append(td_error)
        super().add(state, action, reward, next_state, done)
    
    def sample(self):
        
        """Randomly sample a batch of experiences from memory."""
        
        probabilities = [(p**self.alpha) for p in self.sampling_td_errors]
        
        sum_probs = sum(probabilities)
        
        probabilities = [(p/sum_probs) for p in probabilities]
        
        experience_indexes = np.random.choice(len(self.memory), size=self.batch_size, p=probabilities)
        
        experiences = []
        
        for index in experience_indexes:
            experiences.append(self.memory[index])
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
            
        return (states, actions, rewards, next_states, dones, experience_indexes)

    def update_td_error(self, index, state, action, reward, next_state, done, td_error):
        
        if td_error < 0:
            td_error = - td_error
        
        e = self.experience(state, action, reward, next_state, done)
        self.memory[index] = e
        self.sampling_td_errors[index] = td_error