import io
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from skimage import io, transform
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
import argparse








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
   



# input image
cwd = os.getcwd()
PATH = os.path.join(cwd,"checkpoint_densenettrained.pth")
model_id = 3
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
elif model_id == 3:
    checkpoint = torch.load(PATH)
    net = models.densenet161(pretrained=False)
    num_ftrs = net.classifier.in_features
    net.classifier = nn.Linear(num_ftrs, 2)
    net.load_state_dict(checkpoint)
    finalconv_name = 'features'

net.eval()

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])


liverDataset = LiverDataset(csv_file='testdata.csv',transform = preprocess)

# response = requests.get(IMG_URL)
def run(image, label, imagename, outname, outnamemask):

    img_variable = Variable(image.unsqueeze(0))
    logit = net(img_variable)

    # download the imagenet category list
    classes = {0: 'no_liver', 1: 'liver'}

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()


    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    #print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
    img = cv2.imread(imagename)
    height, width, _ = img.shape
    #heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_BONE)
    heatmap = cv2.resize(CAMs[0],(width, height))
    _, heatmap = cv2.threshold(heatmap,200,255,cv2.THRESH_BINARY)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
    heatmap[:,:,:2] = 0
    #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_AUTUMN)
    result = cv2.addWeighted(heatmap,1.0,img,1.0,0) 
    cv2.imwrite(outname, result)
    cv2.imwrite(outnamemask, heatmap)
liverFrame = pd.read_csv('testdata.csv')
#for i in range(len(liverFrame)):
for i in range(1553,1554):
    im,lb = liverDataset[i]
    imagename = liverFrame['dir'][i]
    names = imagename.split('/')
    filename = ""
    for x in names[-3:-1]:
        filename += x+"/"
    #filename += names[-1]
    outnamemask = "/scratch/sgm400/RADAI/data/test256/outmask/"
    outname = "/scratch/sgm400/RADAI/data/test256/out/"
    #outnamemask = os.path.join(args.data,"test256/outmask")
    #outname = os.path.join(args.data,"test256/out")
    if not os.path.exists(outname):
        os.makedirs(outname)
    if not os.path.exists(outnamemask):
        os.makedirs(outnamemask)
    run(im,lb,imagename, outname+names[-1],outnamemask+names[-1])