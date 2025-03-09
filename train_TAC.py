import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from collections import deque
import random
from ElectricityMarketEnv import ElectricityMarketEnv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten_state(state):
    """
Flatten the state dictionary into a single array.

    Args:
        state (dict): The state dictionary containing keys 'soc', 'demand', 'price', 'renewable', 'season', and 'time'.

    Returns:
        np.ndarray: A flattened array containing all the state information.
    """
    keys = ['soc', 'demand', 'price', 'renewable', 'season', 'time']
    return np.concatenate([state[key].flatten() for key in keys], axis=-1)

# --- Replay Buffer that stores sequences ---
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for storing sequences of states.
    """
    def __init__(self, capacity, history_length, state_dim):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.history_length = history_length
        self.state_dim = state_dim

    def push(self, state_seq, action, reward, next_state_seq, done):
        reward = float(reward)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state_seq, action, reward, next_state_seq, done)
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_seqs, actions, rewards, next_state_seqs, dones = zip(*batch)
        state_seqs = np.stack(state_seqs)
        actions = np.stack(actions)
        rewards = np.array([r if np.isscalar(r) else np.array(r).item() for r in rewards], dtype=np.float32).reshape(-1, 1)
        next_state_seqs = np.stack(next_state_seqs)
        dones = np.array([d if np.isscalar(d) else np.array(d).item() for d in dones], dtype=np.float32).reshape(-1, 1)
        return state_seqs, actions, rewards, next_state_seqs, dones




    def __len__(self):
        return len(self.buffer)


class TransformerStateEncoder(nn.Module):
    def __init__(self, input_dim, d_model, history_length, nhead, num_layers, dropout=0.1):
        super(TransformerStateEncoder, self).__init__()
        # Project the raw state vector to d_model dimensions.
        self.input_proj = nn.Linear(input_dim, d_model)
        # Learnable positional embeddings for the sequence.
        self.pos_embedding = nn.Parameter(torch.zeros(1, history_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # -> (batch, seq_len, d_model)
        x = x + self.pos_embedding  # add positional encoding
        # Transformer expects shape: (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        encoded = self.transformer_encoder(x)  # -> (seq_len, batch, d_model)
        # We take the output of the final token as the state representation.
        encoded = encoded[-1]  # shape: (batch, d_model)
        return encoded

# --- Actor Network ---
class Actor(nn.Module):
    def __init__(self, state_encoder, d_model, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.state_encoder = state_encoder
        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Output mean and log_std for the Gaussian policy.
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state_seq):
        # state_seq shape: (batch, seq_len, input_dim)
        encoded = self.state_encoder(state_seq)
        x = self.fc(encoded)
        mean = self.mean(x)
        log_std = self.log_std(x)
        # Clamp for numerical stability.
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

# --- Critic (Q-Network) ---
class QNetwork(nn.Module):
    def __init__(self, state_encoder, d_model, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.state_encoder = state_encoder
        self.fc = nn.Sequential(
            nn.Linear(d_model + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state_seq, action):
        # state_seq: (batch, seq_len, input_dim); action: (batch, action_dim)
        encoded = self.state_encoder(state_seq)
        x = torch.cat([encoded, action], dim=-1)
        q_value = self.fc(x)
        return q_value


class SACAgent:
    def __init__(self, state_dim, action_dim, history_length, d_model=64, nhead=4, num_layers=2,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, gamma=0.99, tau=0.005, device=device):
        self.history_length = history_length
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Actor network and its transformer encoder
        self.actor_encoder = TransformerStateEncoder(input_dim=state_dim, d_model=d_model,
                                                       history_length=history_length, nhead=nhead, num_layers=num_layers).to(device)
        self.actor = Actor(self.actor_encoder, d_model, action_dim).to(device)
        
        # Two Critic networks and their transformer encoders.
        self.critic_encoder1 = TransformerStateEncoder(input_dim=state_dim, d_model=d_model,
                                                         history_length=history_length, nhead=nhead, num_layers=num_layers).to(device)
        self.critic1 = QNetwork(self.critic_encoder1, d_model, action_dim).to(device)
        
        self.critic_encoder2 = TransformerStateEncoder(input_dim=state_dim, d_model=d_model,
                                                         history_length=history_length, nhead=nhead, num_layers=num_layers).to(device)
        self.critic2 = QNetwork(self.critic_encoder2, d_model, action_dim).to(device)
        
        # Target networks for critics
        self.target_critic_encoder1 = TransformerStateEncoder(input_dim=state_dim, d_model=d_model,
                                                                history_length=history_length, nhead=nhead, num_layers=num_layers).to(device)
        self.target_critic1 = QNetwork(self.target_critic_encoder1, d_model, action_dim).to(device)
        self.target_critic_encoder2 = TransformerStateEncoder(input_dim=state_dim, d_model=d_model,
                                                                history_length=history_length, nhead=nhead, num_layers=num_layers).to(device)
        self.target_critic2 = QNetwork(self.target_critic_encoder2, d_model, action_dim).to(device)
        
        self.target_critic_encoder1.load_state_dict(self.critic_encoder1.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic_encoder2.load_state_dict(self.critic_encoder2.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers.
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()) +
            list(self.critic_encoder1.parameters()) + list(self.critic_encoder2.parameters()),
            lr=critic_lr
        )
        # Temperature (entropy) parameter.
        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -action_dim

    def select_action(self, state_seq, evaluate=False):
        state_seq = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)
        mean, log_std = self.actor(state_seq)
        std = log_std.exp()
        if evaluate:
            action = mean
        else:
            # Reparameterization trick.
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()

            action = torch.tanh(x_t)
        return action.cpu().detach().numpy()[0]

    def update(self, replay_buffer, batch_size):
        state_seqs, actions, rewards, next_state_seqs, dones = replay_buffer.sample(batch_size)
        state_seqs = torch.FloatTensor(state_seqs).to(self.device)       # (B, L, state_dim)
        actions = torch.FloatTensor(actions).to(self.device)             # (B, action_dim)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # (B, 1)
        next_state_seqs = torch.FloatTensor(next_state_seqs).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ----- Critic update -----
        with torch.no_grad():
            next_mean, next_log_std = self.actor(next_state_seqs)
            next_std = next_log_std.exp()
            normal = torch.distributions.Normal(next_mean, next_std)
            x_t = normal.rsample()
            next_action = torch.tanh(x_t)
            log_prob = normal.log_prob(x_t) - torch.log(1 - next_action.pow(2) + 1e-7)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
            target_q1 = self.target_critic1(next_state_seqs, next_action)
            target_q2 = self.target_critic2(next_state_seqs, next_action)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * log_prob
            target_value = rewards + (1 - dones) * self.gamma * target_q

        current_q1 = self.critic1(state_seqs, actions)
        current_q2 = self.critic2(state_seqs, actions)
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ----- Actor update -----
        mean, log_std = self.actor(state_seqs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        q1_pi = self.critic1(state_seqs, action)
        q2_pi = self.critic2(state_seqs, action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = (self.log_alpha.exp() * log_prob - min_q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ----- Temperature (alpha) update -----
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # ----- Soft update of target networks -----
        self.soft_update(self.critic1, self.target_critic1, self.critic_encoder1, self.target_critic_encoder1)
        self.soft_update(self.critic2, self.target_critic2, self.critic_encoder2, self.target_critic_encoder2)
        
        return critic_loss.item(), actor_loss.item(), alpha_loss.item()

    def soft_update(self, critic, target_critic, encoder, target_encoder):
        for param, target_param in zip(critic.parameters(), target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(encoder.parameters(), target_encoder.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def train():
    """
    Train the SAC agent on the ElectricityMarketEnv environment.
    """
    # Hyperparameters.
    num_episodes = 10         # total training episodes
    max_steps = 8760         # steps per episode (as in your environment)
    history_length = 10          # length of the state history fed into the transformer
    batch_size = 64 # batch size for training
    replay_buffer_capacity = 100000
    state_dim = 6                 # flattened state dimension (soc, demand, price, renewable, season, time)
    action_dim = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create your custom environment (make sure ElectricityMarketEnv is imported/defined).
    env = ElectricityMarketEnv()
    
    # Initialize replay buffer.
    replay_buffer = ReplayBuffer(replay_buffer_capacity, history_length, state_dim)
    
    # Initialize SAC agent.
    agent = SACAgent(state_dim, action_dim, history_length, device=device)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = flatten_state(obs)
        # Initialize the history with the first state repeated.
        state_history = deque([state.copy() for _ in range(history_length)], maxlen=history_length)
        episode_reward = 0
        
        for step in range(max_steps):
            if step % 100 == 0:
                print(f"Episode: {episode}, Step: {step}, Reward: {episode_reward}")
            # Prepare current state sequence.
            state_seq = np.array(state_history) 
            # Select action using the actor network.
            action = agent.select_action(state_seq)
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = flatten_state(next_obs)
            # Create the next state sequence.
            next_state_history = state_history.copy()
            next_state_history.append(next_state)
            next_state_seq = np.array(next_state_history)
            # Store transition in the replay buffer.
            replay_buffer.push(state_seq, action, reward, next_state_seq, done)
            
            state_history.append(next_state)
            state = next_state
            episode_reward += reward
            
            # Update agent if enough data is available.
            if len(replay_buffer) > batch_size:
                critic_loss, actor_loss, alpha_loss = agent.update(replay_buffer, batch_size)
            
            if done:
                break
        
        print(f"Episode: {episode}, Reward: {episode_reward}")
        # Save the model every episode.
        save_model(agent, filename=f"./steps_{max_steps}/sac_agent_episode_{episode}.pth")
    # Save the final model.
    save_model(agent, filename="./steps_{max_steps}/sac_agent_final.pth")
def save_model(agent, filename="sac_agent.pth"):
    """
    Save the model parameters to a file.
    Args:
        agent (SACAgent): The SAC agent whose model parameters will be saved.
        filename (str): The path to the file where the model parameters will be saved. Default is "sac_agent.pth".
    """
    checkpoint = {
        'actor_state_dict': agent.actor.state_dict(),
        'actor_encoder_state_dict': agent.actor_encoder.state_dict(),
        'critic1_state_dict': agent.critic1.state_dict(),
        'critic_encoder1_state_dict': agent.critic_encoder1.state_dict(),
        'critic2_state_dict': agent.critic2.state_dict(),
        'critic_encoder2_state_dict': agent.critic_encoder2.state_dict(),
        'target_critic1_state_dict': agent.target_critic1.state_dict(),
        'target_critic_encoder1_state_dict': agent.target_critic_encoder1.state_dict(),
        'target_critic2_state_dict': agent.target_critic2.state_dict(),
        'target_critic_encoder2_state_dict': agent.target_critic_encoder2.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
        'alpha_optimizer_state_dict': agent.alpha_optimizer.state_dict(),
        'log_alpha': agent.log_alpha
    }
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")

def load_model(agent, filename="sac_agent.pth"):
    """
    Load the model parameters from a file.

    Args:
        agent (SACAgent): The SAC agent to load the model into.
        filename (str): The path to the file containing the model parameters. Default is "sac_agent.pth".
    """
    checkpoint = torch.load(filename, map_location=agent.device)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.actor_encoder.load_state_dict(checkpoint['actor_encoder_state_dict'])
    agent.critic1.load_state_dict(checkpoint['critic1_state_dict'])
    agent.critic_encoder1.load_state_dict(checkpoint['critic_encoder1_state_dict'])
    agent.critic2.load_state_dict(checkpoint['critic2_state_dict'])
    agent.critic_encoder2.load_state_dict(checkpoint['critic_encoder2_state_dict'])
    agent.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
    agent.target_critic_encoder1.load_state_dict(checkpoint['target_critic_encoder1_state_dict'])
    agent.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
    agent.target_critic_encoder2.load_state_dict(checkpoint['target_critic_encoder2_state_dict'])
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
    # Update the log_alpha parameter.
    agent.log_alpha.data.copy_(checkpoint['log_alpha'].data)
    print(f"Model loaded from {filename}")

if __name__ == "__main__":
    print(f'device: {device}')
    train()
