from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from PIL import Image
import warnings
import argparse
warnings.filterwarnings("ignore")




parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', default = '/scratch/sgm400/RADAI/data/'
                    help='path to dataset')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch', default=0, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batchsize', default=32, type=int, metavar='N',
                    help='Batch size for training')
parser.add_argument('--imgsize', default=224, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--classes', default=2, type=int, metavar='N',
                    help='number of classes in the dataset')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--feature_extract', dest='feature_extract', action='store_true',
                    help='do feature extracting')




def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class LiverDataset(Dataset):
    def __init__(self, csv_file, transform = None):
        self.liverFrame = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.liverFrame)
    def __getitem__(self, idx):
        img_name = self.liverFrame['dir'][idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        image = image[:3,:,:]
        label = self.liverFrame['label'][idx]
        return image,label


dir1 = os.path.join(args.data,"train256/images_pngs_noliver")
dir2 = os.path.join(args.data,"train256/images_pngs_liver")
dir3 = os.path.join(args.data,"test256/images_pngs")
raw_data = {'dir':[],'label':[]}
raw_data_test = {'dir':[],'label':[]}
for filename in os.listdir(dir1):
    file = os.path.join(dir1,filename)
    if file[-3:] == "png":
        raw_data['dir'].append(file)
        raw_data['label'].append(0)
for filename in os.listdir(dir2):
    file = os.path.join(dir2,filename)
    if file[-3:] == "png":
        raw_data['dir'].append(file)
        raw_data['label'].append(1)
for filename in os.listdir(dir3):
    file = os.path.join(dir3,filename)
    if file[-3:] == "png":
        raw_data_test['dir'].append(file)
        raw_data_test['label'].append(1)
df = pd.DataFrame(raw_data, columns = ['dir','label'])
df.to_csv('data.csv')
dftest = pd.DataFrame(raw_data_test, columns = ['dir','label'])
dftest.to_csv('testdata.csv')

# prepare data
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
)

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(args.imgsize),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

liverDataset = LiverDataset(csv_file='data.csv',transform = transform_train)
trainLoader = DataLoader(liverDataset, batch_size=args.batchsize, shuffle=True, num_workers=2)
classes = {0: 'no_liver', 1: 'liver'}
#net = models.resnet18(pretrained=True)
net = models.densenet161(pretrained=True)
set_parameter_requires_grad(net, args.feature_extract)
#num_ftrs = net.fc.in_features
#net.fc = nn.Linear(num_ftrs, num_classes)
num_ftrs = net.classifier.in_features
net.classifier = nn.Linear(num_ftrs, args.classes)
print(net)
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


epoch       = args.epoch
for e in range(arg.epochs):
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


