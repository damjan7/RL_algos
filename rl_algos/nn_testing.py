# create 2 simple nn's and freeze weights of one


import torch
from torch import nn
import torch.nn.functional as F
from torch import optim


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size  # 5 features
        self.output_size = output_size  # =3
        self.hidden = 25

        self.fc1 = nn.Linear(input_size, self.hidden, bias=True)
        self.fc2 = nn.Linear(self.hidden, self.hidden, bias=True)
        self.fc3 = nn.Linear(self.hidden, self.output_size, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


main_nn = Net(5, 3)
target_nn = Net(5,3)

criterion = nn.MSELoss()
optimizer = optim.Adam(main_nn.parameters())

test_vec = torch.tensor([1, 5, 9, 2.12, 0.25]).view(1, 5)

def train_step():
    pass
