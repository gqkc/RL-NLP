import torch
import torch.nn as nn
import torch.optim as optim

from models.rl_basic import PolicyGRUWord


class REINFORCE:
    def __init__(self, hidden_size, word_emb_size, action_space, gamma=0.9, lr=1e-2):
        self.model = PolicyGRUWord(num_tokens=action_space, word_emb_size=word_emb_size, hidden_size=hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state, valid_actions):
        m, value = self.model(state.text, state.img, list(valid_actions.values()))
        valid_action = m.sample()
        action=torch.tensor(valid_actions[valid_action.item()]).view(1)
        return action.item(), m.log_prob(valid_action).view(1), value

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        mse = nn.MSELoss()
        for r in self.model.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).float()
        for log_prob, R, value in zip(self.model.saved_log_probs, returns, self.model.values):
            policy_loss.append(-log_prob * (R - value))
            ms = mse(value, R)
            policy_loss.append(ms.view(1))
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.model.rewards[:]
        del self.model.saved_log_probs[:]
        del self.model.values[:]