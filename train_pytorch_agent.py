import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_review_env import CodeReviewAction
from code_review_env.server.environment import CodeReviewEnvironment
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

def extract_state(obs):

    state = [
        obs.churn_score / 100.0,
        obs.complexity_score / 100.0,
        obs.todo_score / 100.0,
        obs.recency_score / 100.0,
        obs.files_flagged / max(1.0, float(obs.review_budget)),
        obs.review_budget / 20.0
    ]
    return torch.FloatTensor(state).unsqueeze(0)


def main():
    input_dim = 6
    hidden_dim = 32
    output_dim = 2
    learning_rate = 0.01
    
    policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    num_episodes = 50
    print(f"Initializing PyTorch REINFORCE Agent natively on CodeReviewEnv for {num_episodes} episodes...")
    
    env = CodeReviewEnvironment()
    
    import random
    difficulties = ["easy", "medium", "hard"]
    for episode in range(num_episodes):
        difficulty = random.choice(difficulties)
        obs = env.reset(difficulty=difficulty)
        
        saved_log_probs = []
        rewards = []
        
        while not obs.done:
            state = extract_state(obs)
            

            probs = policy_net(state)
            m = Categorical(probs)
            

            action = m.sample()
            saved_log_probs.append(m.log_prob(action))
            
            decision = "flag" if action.item() == 1 else "skip"
            
            obs = env.step(CodeReviewAction(decision=decision))
            rewards.append(obs.reward)


        gamma = 0.99
        R = 0
        policy_loss = []
        returns = []
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)

        if len(returns) > 1 and returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        for log_prob, R_val in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * R_val)
            
        optimizer.zero_grad()
        if policy_loss:
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            optimizer.step()
        
        if (episode + 1) % 10 == 0 or episode == 0:
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Repo: {obs.repo_name} | "
                  f"Loss: {policy_loss.item() if policy_loss else 0:.2f} | "
                  f"Reward: {sum(rewards):.1f} | "
                  f"F1: {obs.f1_score:.2f} (P: {obs.precision:.2f}, R: {obs.recall:.2f})")
            
    print("\nTraining complete! PyTorch internal weights updated.")
    
if __name__ == "__main__":
    main()
