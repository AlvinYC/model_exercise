import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os

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


#net = Net1()
#net = Net2()
net = Net3()
#net = NetCEI201()
#net = VGG('VGG16')
print(net)
net.cuda()

data_batch_size = 128
start_time = time.time()
if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset  = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    ## reduce training size -Í begin
    num_train = len(trainset)
    indices   = list(range(num_train))
    split     = int(np.floor(0.5 * num_train))
    np.random.seed(10)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    ## reduce training size - end
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=data_batch_size, shuffle=True, num_workers=2)
    #trainloader = torch.utils.data.DataLoader(trainset,batch_size=data_batch_size, sampler=train_sampler, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=data_batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.001)

    #plot image to check content
    images, _ = next(iter(trainloader))
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(torchvision.utils.make_grid(images/2+0.5, nrow=16).permute((1, 2, 0)))
    plt.show()

    train = True
    #train = False
    if train:
        for epoch in range(10):
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                #inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                output = net(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                #running_loss += loss.data[0]
                running_loss += loss.data.item()
                
                show_iter = 200
                if i % show_iter == (show_iter-1):
                    print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / show_iter))
                    running_loss = 0.0

        print('Finished Training')

        torch.save(net.state_dict(), './cifar10-Net-model')
print("--- %s seconds ---" % (time.time() - start_time))  
