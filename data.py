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
warnings.filterwarnings("ignore")

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


# "/Users/srinidhigoud/Desktop/RADAI/data/radai_challenge/train256/images_pngs_noliver"
cwd = os.getcwd()
# dir1 = os.path.join(cwd,"data/radai_challenge/train256/images_pngs_noliver")
# dir2 = os.path.join(cwd,"data/radai_challenge/train256/images_pngs_liver")
dir1 = "/scratch/sgm400/RADAI/data/radai_challenge/train256/images_pngs_noliver"
dir2 = "/scratch/sgm400/RADAI/data/radai_challenge/train256/images_pngs_liver"
# print(dir1)
raw_data = {'dir':[],'label':[]}
for filename in os.listdir(dir1):
    file = os.path.join(dir1,filename)
    raw_data['dir'].append(file)
    raw_data['label'].append(0)
for filename in os.listdir(dir2):
    file = os.path.join(dir2,filename)
    raw_data['dir'].append(file)
    raw_data['label'].append(1)
df = pd.DataFrame(raw_data, columns = ['dir','label'])
df.to_csv('data.csv')

# prepare data
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
)

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
num_classes = 2
BATCH_SIZE      = 32
IMG_SIZE        = 224
epoch           = 0
epochs = 10
feature_extract = True

liverDataset = LiverDataset(csv_file='data.csv',transform = transform_train)
trainLoader = DataLoader(liverDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
classes = {0: 'no_liver', 1: 'liver'}
net = models.resnet18(pretrained=True)
set_parameter_requires_grad(net, feature_extract)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, num_classes)
print(net)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
finalconv_name = 'features'

criteria = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.005, momentum = 0.5)
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

# #define validation function
# def validation (model, loader, criterion, gpu):
#     model.eval()
#     valid_loss = 0
#     valid_correct = 0
#     for valid, y_valid in iter(loader):
#         if gpu:
#             valid, y_valid = valid.to('cuda'), y_valid.to('cuda')
#         output = model.forward(valid)
#         valid_loss += criterion(output, y_valid).item()*valid.size(0)
#         equal = (output.max(dim=1)[1] == y_valid.data)
#         valid_correct += torch.sum(equal)#type(torch.FloatTensor)
    
#     epoch_loss = valid_loss / len(validLoader.dataset)
#     epoch_acc = valid_correct.double() / len(validLoader.dataset)
    
#     return epoch_loss, epoch_acc


for e in range(epochs):
    epoch +=1
    print(epoch)
    with torch.set_grad_enabled(True):
        epoch_train_loss, epoch_train_acc = train(net,trainLoader, criteria)
    print("Epoch: {} Train Loss : {:.4f}  Train Accuracy: {:.4f}".format(epoch,epoch_train_loss,epoch_train_acc))
# with torch.no_grad():
#         epoch_val_loss, epoch_val_acc = validation(model, validLoader, criteria, args.gpu)
#     print("Epoch: {} Validation Loss : {:.4f}  Validation Accuracy {:.4f}".format(epoch,epoch_val_loss,epoch_val_acc))

# img,label = liverDataset[10]
# print(len(img))



