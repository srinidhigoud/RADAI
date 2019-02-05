from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import util
from util import *

args = util.get_args()
if(args.generatecsv):
    util.generateCSV(args.data)
liverDataset = util.LiverDataset(csv_file='data.csv',transform = util.transform_train)
trainLoader = DataLoader(liverDataset, batch_size=args.batchsize, shuffle=True, num_workers=2)
classes = {0: 'no_liver', 1: 'liver'}
#net = models.resnet18(pretrained=True)
net = models.densenet161(pretrained=True)
util.set_parameter_requires_grad(net, args.feature_extract)
#num_ftrs = net.fc.in_features
#net.fc = nn.Linear(num_ftrs, num_classes)
num_ftrs = net.classifier.in_features
net.classifier = nn.Linear(num_ftrs, args.classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
finalconv_name = 'features'

criteria = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.5)
def train (model, loader, criterion):
    model.train()
    current_loss = 0
    current_correct = 0
    for train, y_train in iter(loader):
        train, y_train = train.to(device), y_train.to(device)
        optimizer.zero_grad()
        output = model.forward(train)
        _, preds = torch.max(output,1)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        current_loss += loss.item()*train.size(0)
        current_correct += torch.sum(preds == y_train.data)
    epoch_loss = current_loss / len(trainLoader.dataset)
    epoch_acc = current_correct.double() / len(trainLoader.dataset)
        
    return epoch_loss, epoch_acc


epoch = args.epoch
for e in range(args.epochs):
    epoch +=1
    print(epoch)
    with torch.set_grad_enabled(True):
        epoch_train_loss, epoch_train_acc = train(net,trainLoader, criteria)
    print("Epoch: {} Train Loss : {:.4f}  Train Accuracy: {:.4f}".format(epoch,epoch_train_loss,epoch_train_acc))

#Better way to save the chepoints is to keep track of the best performing with respect to test accuracy model and save it
file_name = "checkpoint_densenettrained.pth"
torch.save(net.state_dict(), file_name)
check_path = Path(file_name)
print("File Size: {} K".format(check_path.stat().st_size/10**3))