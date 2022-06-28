import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name= "model.pth"):
        folder_path = './model'
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        
        file_name = os.path.join(folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Qtrainer:

    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.obtimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() #loss function
    
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        # (n, x)

        if len(state) == 1:
            state = torch.unsqueeze(state, dim = 0)
            action = torch.unsqueeze(action, dim = 0)
            reward = torch.unsqueeze(reward, dim = 0)
            next_state = torch.unsqueeze(next_state, dim = 0)

            done = (done, )
        
        # 1: pred Q values with current state

        pred = self.model(state)

        target = pred.clone()

        # 2: Q_new = r + y*(max Q values of next pred)
        # pred.clone()
        # pred[argmax(pred).item()] = Q_new replace the max with the predicted Q

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = Q_new + self.gamma*(torch.max(self.model(next_state[idx])))

            target[idx][torch.argmax(action).item()] = Q_new
        
        self.obtimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.obtimizer.step()
