"""
utils
"""

from __future__ import print_function
import os
import torch
import pandas as pd
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import warnings
import argparse
warnings.filterwarnings("ignore")




def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', metavar='DIR', default ='/scratch/sgm400/RADAI/data/',
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
    parser.add_argument('--generatecsv', dest='generatecsv', action='store_false',
                        help='Generate csv files for data')
    args = parser.parse_args()
    return args


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

def generateCSV(dataDir):
    dir1 = os.path.join(dataDir,"train256/images_pngs_noliver")
    dir2 = os.path.join(dataDir,"train256/images_pngs_liver")
    dir3 = os.path.join(dataDir,"test256/images_pngs")
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
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

