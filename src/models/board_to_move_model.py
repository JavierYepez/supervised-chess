import torch.nn as nn


class BoardToMoveModel(nn.Module):
    def __init__(self):
        super(BoardToMoveModel, self).__init__()
        self.conv1 = nn.Conv2d(14, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 256, 600)
        self.fc2 = nn.Linear(600, 400)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200, 100)
        self.output_from = nn.Linear(100, 64)
        self.output_to = nn.Linear(100, 64)
        self.relu = nn.ReLU()

        # Initialize weights
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.output_from.weight)
        nn.init.xavier_uniform_(self.output_to.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        out_from = self.output_from(x)
        out_to = self.output_to(x)
        return out_from, out_to