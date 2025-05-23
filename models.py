import torch
from torch import nn
import torch.nn.functional as F

from scattering_transform.scattering_transform import ScatteringTransform2d
from scattering_transform.filters import FourierSubNetFilters, SubNet, Morlet
from scattering_transform.reducer import Reducer

import copy


class NFSTRegressor(nn.Module):
    def __init__(self, size, num_scales, num_angles=4, subnet_hidden_sizes=(32, 32), init_morlet=True,
                 reduction='asymm_ang_avg', linear_hiddens=128, symmetric_filters=False):
        super(NFSTRegressor, self).__init__()

        self.size = size
        self.num_scales = num_scales
        self.num_angles = num_angles

        self.subnet = SubNet(num_ins=3, hidden_sizes=subnet_hidden_sizes, num_outs=1, hidden_activation=nn.LeakyReLU)

        self.filters = FourierSubNetFilters(size, num_scales, num_angles, subnet=self.subnet,
                                            symmetric=symmetric_filters, init_morlet=init_morlet, full_rotation=True)
        self.filters.update_filters()

        # get the initial filters state dict
        self.initial_filters_state = copy.deepcopy(self.filters.state_dict())

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

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = super(NFSTRegressor, self).state_dict(destination, prefix, keep_vars)
        state[prefix + 'initial_filters_state'] = self.initial_filters_state
        return state

    def load_state_dict(self, state_dict, strict=True):
        self.initial_filters_state = state_dict.pop('initial_filters_state')
        super(NFSTRegressor, self).load_state_dict(state_dict, strict)


class MSTRegressor(nn.Module):
    def __init__(self, size, num_scales, num_angles=4, reduction='asymm_ang_avg', linear_hiddens=128):
        # This is not the best way to implement. Ideally, we would pre-calculate the MST and then use that as input to the regressor.
        # However, doing it this way ensures consistency with the NFSTRegressor class. It also means that we can use the same
        # training loop for both models.

        super(MSTRegressor, self).__init__()

        self.size = size
        self.num_scales = num_scales
        self.num_angles = num_angles

        self.filters = Morlet(size, num_scales, num_angles)

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
    def __init__(self, circular=False):

        padding_mode = 'circular' if circular else 'zeros'

        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, 11, padding=5, padding_mode=padding_mode)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2a = nn.Conv2d(100, 100, 5, padding=2, padding_mode=padding_mode)
        self.conv2b = nn.Conv2d(100, 100, 5, padding=2, padding_mode=padding_mode)
        self.conv3a = nn.Conv2d(100, 100, 3, padding=1, padding_mode=padding_mode)
        self.conv3b = nn.Conv2d(100, 100, 3, padding=1, padding_mode=padding_mode)
        self.conv3c = nn.Conv2d(100, 100, 3, padding=1, padding_mode=padding_mode)
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