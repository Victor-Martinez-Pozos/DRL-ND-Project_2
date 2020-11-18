from networks import ValueNetwork, SoftQNetwork, PolicyNetwork
from utils import ReplayBuffer

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class SoftActorCritic:

    def __init__(self, action_dim, state_dim, hidden_dim = 256):
        #action_dim = env.action_space.shape[0]
        #state_dim  = env.observation_space.shape[0]
       
        self.value_net = ValueNetwork(state_dim, hidden_dim).to(device)
        self.target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)
        
        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)
            
        self.value_criterion  = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        
        value_lr  = 3e-4
        soft_q_lr = 3e-4
        policy_lr = 3e-4
        
        self.value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr)
        self.soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
        
        
        replay_buffer_size = 1000000
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def act(self, state, train=True):
        if train:
            action = self.policy_net.get_action(state).detach()
        else:
            with torch.no_grad():
                action = self.policy_net.get_action(state).detach()
        return action

    def update(self, batch_size, gamma=0.99,soft_tau=1e-2,):
    
        state, action, reward, next_state, done = self.replay_buffer\
                                                  .sample(batch_size)
    
        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
    
        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        predicted_value    = self.value_net(state)
        new_action, log_prob, epsilon, mean, log_std = self.policy_net\
                                                       .evaluate(state)
    
        
        
        # Training Q Function
        target_value = self.target_value_net(next_state)
        target_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())
    
    
        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()    
        # Training Value Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action),self.soft_q_net2(state, new_action))
        target_value_func = predicted_new_q_value - log_prob
        value_loss = self.value_criterion(predicted_value, target_value_func.detach())
    
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        # Training Policy Function
        policy_loss = (log_prob - predicted_new_q_value).mean()
    
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )