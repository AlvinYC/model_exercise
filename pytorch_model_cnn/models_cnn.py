import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Net1: 1 conv + 3 fc
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        #self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1 = nn.Conv2d(3, 128, 5)
        self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2 = nn.Conv2d(128, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Net2: 2 conv + 3 fc
class Net2(nn.Module):
    def __init__(self): # （3,32,32)
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1, 2) # (20,32,32)
        self.pool = nn.MaxPool2d(2, 2)         # (20,16,16)
        self.conv2 = nn.Conv2d(20, 16, 5,1, 2) # (16,16,16)
        self.fc1 = nn.Linear(16 * 8 * 8, 120)  #
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#  3-layer
#  conv2d(3,20,5) -> pool(2,2) -> conv2d(20,64,5) -> pool(2,2) -> conv2d(64,128,5) -> pool(2,2)
#
class Net3(nn.Module):
    def __init__(self): # （3,32,32)
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(3,  20, 5, 1, 2) # (20,32,32)
        self.pool = nn.MaxPool2d(2, 2)          # (20,16,16)
        self.conv2 = nn.Conv2d(20, 64, 5, 1, 2) # (64,16,16)
        self.conv3 = nn.Conv2d(64,128, 5, 1, 2) # (128,8 ,8)
        self.fc1 = nn.Linear(128 * 4 * 4, 64)  #
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
# NetCEI201: 2 conv + 3 fc
# hands on problem:
# in this network, 
#     i wanna use  6 channel in first convolusion layer
#                 15 channel in second convolusion layer
#                 64 neuron in first fc layer
#                 32 neuron in second fc layer
# how can i design this network

'''
class NetCEI201(nn.Module):
    def __init__(self): # （3,32,32)
        super(NetCEI201, self).__init__()
        self.conv1 = nn.Conv2d(?, ?, 5, 1, 2)  # (?,?,?)
        self.pool = nn.MaxPool2d(2, 2)         # (?,?,?)
        self.conv2 = nn.Conv2d(?, ?, 5, 1, 2)  # (?,?,?)
        self.fc1 = nn.Linear(? * ? * ?, ?)     #
        self.fc2 = nn.Linear(?, ?)
        self.fc3 = nn.Linear(?, ?)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, ? * ? * ?)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x    
'''

    
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        #self.classifier = nn.Linear(8192,2)
        self.classifier = nn.Sequential(
            #nn.Linear(512 * 4 * 4, 4096),
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Dropout(),
            #nn.Linear(4096, 4096),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            #nn.Linear(4096, 10),
            nn.Linear(512, 10),
        )
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.softmax(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)