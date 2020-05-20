import torch
import torch.nn as nn
import torch.optim as optim

from models.rl_basic import PolicyGRUWord, ValueFunctionWord


class REINFORCE:
    def __init__(self, hidden_size, word_emb_size, action_space, gamma=0.9, lr=1e-2):
        self.model = PolicyGRUWord(num_tokens=action_space, word_emb_size=word_emb_size, hidden_size=hidden_size)
        self.value_model = ValueFunctionWord(num_tokens=action_space, word_emb_size=word_emb_size,
                                             hidden_size=hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.model.parameters(), lr=0.008)
        self.gamma = gamma

    def select_action(self, state, valid_actions):
        m, value = self.model(state.text, state.img, list(valid_actions.values()))
        value = self.value_model(state.text, state.img)
        valid_action = m.sample()
        action = torch.tensor(valid_actions[valid_action.item()]).view(1)
        return action.item(), m.log_prob(valid_action).view(1), value

    def finish_episode(self):
        R = 0
        policy_loss = []
        value_loss=[]
        returns = []
        mse = nn.MSELoss()
        for r in self.model.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).float()
        for log_prob, R, value in zip(self.model.saved_log_probs, returns, self.model.values):
            policy_loss.append(-log_prob * (R - value))
            ms = mse(value, R)
            value_loss.append(ms.view(1))
        self.optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        value_loss = torch.cat(value_loss).sum()

        policy_loss.backward(retain_graph=True)
        value_loss.backward()

        self.optimizer.step()
        self.value_optimizer.step()

        del self.model.rewards[:]
        del self.model.saved_log_probs[:]
        del self.model.values[:]
