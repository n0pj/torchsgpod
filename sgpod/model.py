import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as D
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils import imshow, print_info
from tqdm import tqdm
import time, sys, os
from layers import Conv2dbnleaky, Resblock, GlobalAveragePool2d, XBlock, XBlockv2
from torchsummary import summary

torch.backends.cudnn.benchmark=True
# transform = transforms.Compose(
#     [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
print_info("===>Load dataset...", ["cyan", "bold"])
# trainset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
# testset = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

transform = transforms.Compose(
    [transforms.Resize((256, 256)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
full_dataset = torchvision.datasets.ImageFolder(root="rawdata/", transform=transform)
train_size = int(0.8*len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)

classes = len(os.listdir("rawdata/"))

# classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck", )

# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# imshow(torchvision.utils.make_grid(images))
# print(" ".join(classes[labels[j]] for j in range(4)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 32, 2, stride=2, padding=1)
        # self.conv1 = Conv2dbnleaky(3, 32, 2, stride=2)
        # self.conv2 = Conv2dbnleaky(3, 32, 1)
        self.res1 = XBlock(3, 32, 2)
        self.res2 = XBlock(32, 64, 2)
        self.res3 = XBlock(64, 128, 4)
        self.res4 = XBlock(128, 256, 8)
        self.res5 = XBlock(256, 512, 4)
        # self.res1 = Resblock(3, 32, 2)
        # self.res2 = Resblock(32, 64, 4)
        # self.res3 = Resblock(64, 128, 2)
        # self.res4 = Resblock(128, 256, 2)
        # self.res5 = Resblock(256, 512, 2)
        self.gap1 = GlobalAveragePool2d(512, 10, 1)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.gap1(x)
        # x = x.view(-1, 64*8*8)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

print_info("===>Create Network...", ["cyan", "bold"])

# net = Net()
from torchvision import models
et = time.time()
net = models.vgg16(pretrained=True)
net.classifier[6] = nn.Linear(4096, classes)
# net.load_state_dict(torch.load("weights/vloss0.5725-vacc0.8790-loss0.1197-acc0.9661.pth"))
device = torch.device(0)
net.to(device)
print(time.time() - et)
print(net)

print(sys.argv)
epochs = int(sys.argv[1])
print(epochs)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.00015)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs*0.5, epochs*0.8], gamma=0.1)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95**epoch)
print(optimizer)

from utils import progress

print_info("===>Training start...", ["cyan", "bold"])

net.train()
for epoch in range(epochs):
    running_loss = 0.0
    running_tloss = 0.0
    correct = 0
    total = 0
    scheduler.step()
    optemp = optimizer.state_dict()
    print_info("lr = {}".format(optemp["param_groups"][0]["lr"]), ["cyan", "bold"])
    estimate = time.time()
    progress(0, len(trainloader), 0, 0, e=estimate, epoch=epoch, epochs=epochs)
    for i, data in enumerate(trainloader):
        optimizer.zero_grad()
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        estimate2 = time.time() - estimate

        _, predicted = torch.max(outputs.data, 1)
        total += labels.to(0).size(0)
        correct += (predicted == labels.to(0)).sum().item()
        acc = correct/total

        if i+1 != len(trainloader):
            progress(i+1, len(trainloader), running_loss/(i+1), acc, e=estimate2, epochs=epochs)
        else:
            correct = 0
            total = 0
            temp_vloss = 0

            net.eval()
            with torch.no_grad():
                for idx, data in enumerate(testloader):
                    # inputs, labels = iter(testloader).next()[0].to(device), iter(testloader).next()[1].to(device)
                    inputs, labels = data[0].to(device), data[1].to(device)

                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    running_tloss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    vacc = correct/total

                progress(i+1, len(trainloader), running_loss/(i+1), acc, vloss=running_tloss/(idx+1) , vacc=vacc, e=estimate2, epochs=epochs)
                
                if epoch != 0:

                    if (epoch+1)%3 == 0:
                        print_info("===>Model saving...", ["cyan", "bold"])
                        torch.save(net.state_dict(), "weights/vloss{:.4f}-vacc{:.4f}-loss{:.4f}-acc{:.4f}.pth".format(running_tloss/(idx+1) , vacc, running_loss/(i+1), acc))
                        print_info("vloss{:.4f}-vacc{:.4f}-loss{:.4f}-acc{:.4f}.pth".format(running_tloss/(idx+1) , vacc, running_loss/(i+1), acc))
                        print_info("===>Saving done...", ["cyan", "bold"])

                else:
                    pass

print_info("===>Training complete...\n", ["cyan", "bold"])

print_info("===>Model saving...\n", ["cyan", "bold"])
torch.save(net.state_dict(), "sgpodres.pth")
print_info("===>Saving done...\n", ["cyan", "bold"])
    # #------ここから-------#
    # vis.line(X=np.array([epoch]), Y=np.array([running_loss/(i+1)]), win='loss', name='train_loss', update='append')
    # vis.line(X=np.array([epoch]), Y=np.array([acc]), win='acc', name='train_acc', update='append')
    # vis.line(X=np.array([epoch]), Y=np.array([running_tloss/(idx+1)]), win='loss', name='test_loss', update='append')
    # vis.line(X=np.array([epoch]), Y=np.array([vacc]), win='acc', name='test_acc', update='append')
    # #------ここまで-------#

# correct = 0
# total = 0
# running_tloss = 0.0
# estimate = time.time()
# progress(0, len(trainloader), 0, estimate, epoch=0)
# with torch.no_grad():
#     for i, data in enumerate(testloader):
#         optimizer.zero_grad()
#         inputs, labels = data
#         inputs = inputs.to(0)
#         labels = labels.to(0)
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         optimizer.step()
#         running_tloss += loss.item()

#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.to(0).size(0)
#         correct += (predicted == labels.to(0)).sum().item()
#         acc = correct/total

#         estimate2 = time.time() - estimate
#         progress(i+1, len(testloader), running_tloss/(i+1), acc, e=estimate2)

    