import torch
from torch import nn
import torch.nn.functional as F

from scattering_transform.scattering_transform import *
from scattering_transform.filters import *


class NFSTRegressor(nn.Module):
    def __init__(self, size, num_scales, num_angles=4, subnet_hidden_sizes=(32, 32), init_morlet=True,
                 reduction='asymm_ang_avg', linear_hiddens=128):
        super(NFSTRegressor, self).__init__()

        self.size = size
        self.num_scales = num_scales
        self.num_angles = num_angles

        self.subnet = SubNet(num_ins=3, hidden_sizes=subnet_hidden_sizes, num_outs=1, activation=nn.LeakyReLU)

        self.filters = FourierSubNetFilters(size, num_scales, num_angles, subnet=self.subnet, symmetric=False,
                                            init_morlet=init_morlet, full_rotation=True)
        self.filters.update_filters()

        self.st = ScatteringTransform2d(self.filters)
        self.reducer = Reducer(self.filters, reduction, filters_3d=False)

        self.num_outputs = self.reducer.num_outputs

        self.regressor = nn.Sequential(
            nn.Linear(self.num_outputs, linear_hiddens),
            nn.ReLU(),
            nn.Linear(linear_hiddens, 1)
        )

    def forward(self, x):
        self.filters.update_filters()
        x = self.st(x)
        x = self.reducer(x).squeeze(1)
        return self.regressor(x)

    def to(self, device):
        super(NFSTRegressor, self).to(device)
        self.filters.to(device)
        self.st.to(device)
        self.device = device
        return self


class MSTRegressor(nn.Module):
    def __init__(self, size, num_scales, num_angles=4, subnet_hidden_sizes=(32, 32),
                 reduction='asymm_ang_avg', linear_hiddens=128):
        super(MSTRegressor, self).__init__()

        self.size = size
        self.num_scales = num_scales
        self.num_angles = num_angles

        self.filters = Morlet(size, num_scales, num_angles)
        self.filters.update_filters()

        self.st = ScatteringTransform2d(self.filters)
        self.reducer = Reducer(self.filters, reduction, filters_3d=False)

        self.num_outputs = self.reducer.num_outputs

        self.regressor = nn.Sequential(
            nn.Linear(self.num_outputs, linear_hiddens),
            nn.ReLU(),
            nn.Linear(linear_hiddens, 1)
        )

    def forward(self, x):
        x = self.st(x)
        x = self.reducer(x).squeeze(1)
        return self.regressor(x)

    def to(self, device):
        super(MSTRegressor, self).to(device)
        self.filters.to(device)
        self.st.to(device)
        self.device = device
        return self


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, 11, padding=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2a = nn.Conv2d(100, 100, 5, padding=2)
        self.conv2b = nn.Conv2d(100, 100, 5, padding=2)
        self.conv3a = nn.Conv2d(100, 100, 3, padding=1)
        self.conv3b = nn.Conv2d(100, 100, 3, padding=1)
        self.conv3c = nn.Conv2d(100, 100, 3, padding=1)
        self.fc1 = nn.Linear(6400, 4000)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4000, 4000)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4000, 1000)
        self.fc4 = nn.Linear(1000, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = F.relu(self.conv3c(x))
        x = self.pool(x)
        x = x.view(-1, 100 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def to(self, device):
        super(CNN, self).to(device)
        self.device = device
        return self