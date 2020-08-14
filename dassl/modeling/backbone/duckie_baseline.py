"""
Reference

https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/tree/master/M3SDA
"""
import torch.nn as nn
from torch.nn import functional as F

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

class FeatureExtractor(Backbone):

    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()
        flat_size = 32 * 2 * 4
        defaults.update((k, params[k]) for k in defaults.keys() & params.keys())
        params = defaults

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self._out_features = 128*15*20
        self.dropout = nn.Dropout(.5)

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 60 and W == 80, \
            'Input to network must be 60x80, ' \
            'but got {}x{}'.format(H, W)

    def forward(self, x):
        """Forward pass x."""
        x = self.bn1(self.lr(self.conv1(x)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        return x


@BACKBONE_REGISTRY.register()
def duckie_baseline(**kwargs):
    """
    """
    return FeatureExtractor(2, 1)