import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils_reg import *
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import os
from collections import OrderedDict
from itertools import cycle
import cv2
import pdb
from pdb import set_trace as bp
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm import tqdm

#writer = SummaryWriter('runs/da_valid_file_train_tail_1')
torch.autograd.set_detect_anomaly(True)
device_ids=[2,3]
device_1 = f'cuda:{device_ids[0]}'
device_2= f'cuda:{device_ids[1]}'
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Datasets(Dataset):
    def __init__(self, datasetA, datasetB):
        self.datasetA = datasetA
        self.datasetB = datasetB
        
    def __getitem__(self, index):
        (xA,lA) = (self.datasetA[index][0],self.datasetA[index][1])
        (xB,lB) = (self.datasetB[index][0],self.datasetB[index][1])
        return (xA,lA), (xB,lB)
    
    def __len__(self):
        return len(self.datasetA)

class Unlabeled_Datasets(Dataset):
    def __init__(self, datasetA, datasetB):
        self.datasetA = datasetA
        self.datasetB = datasetB
        
    def __getitem__(self, index):
        xA = self.datasetA[index]
        xB = self.datasetB[index]
        return (xA,xB)
    
    def __len__(self):
        return len(self.datasetA)

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = sorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

transform = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
EO_file_pth = '/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/train_images/ensemble_train_EO'
SAR_file_pth = '/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/train_images/ensemble_train_SAR'

EO_unlabeled = '/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/NTIRE2021_Class_valid_images_EO'
SAR_unlabeled = '/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/NTIRE2021_Class_valid_images_SAR'

train_unlabeled_EO = CustomDataSet(main_dir=EO_unlabeled,transform=transform)
train_unlabeled_SAR = CustomDataSet(main_dir=SAR_unlabeled,transform=transform)

train_data_EO = torchvision.datasets.ImageFolder(root=EO_file_pth,transform=transform)
train_data_SAR = torchvision.datasets.ImageFolder(root=SAR_file_pth,transform=transform)

train_dataset_size_EO = train_data_EO.__len__()
train_dataset_size_SAR = train_data_SAR.__len__()

train_dataset = Datasets(train_data_EO,train_data_SAR)
unlabeled_dataset = Unlabeled_Datasets(train_unlabeled_EO,train_unlabeled_SAR)
################################ weighted random sampler for training for EO domain ####################################################
y_train_indices_EO = len(train_data_EO)

y_train_EO = [train_data_EO.targets[i] for i in range(y_train_indices_EO)]

class_sample_count = np.array(
    [len(np.where(y_train_EO == t)[0]) for t in np.unique(y_train_EO)])

weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in y_train_EO])
samples_weight = torch.from_numpy(samples_weight)

sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),replacement=True)
train_loader = data.DataLoader(train_dataset, batch_size=32, sampler=sampler)
#train_loader = data.DataLoader(train_dataset, batch_size=64, num_workers=5)
unlabeled_train_loader = data.DataLoader(unlabeled_dataset, batch_size=32)
#####################################################################################################################################################
def train():
    model_EO = models.efficientnet_b0(pretrained=True)
    num_ftrs_EO = model_EO.classifier[1].in_features
    model_EO.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(num_ftrs_EO, 10))
    
    for param in model_EO.parameters():
        param.requires_grad = True

    model_SAR = models.efficientnet_b0(pretrained=True)
    num_ftrs_SAR = model_SAR.classifier[1].in_features
    model_SAR.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(num_ftrs_SAR, 10))

    for param in model_SAR.parameters():
        param.requires_grad = True

    model_EO = torch.load('EO_cross_domain_EffB0.pth')
    model_SAR = torch.load('SAR_cross_domain_EffB0.pth')

    alpha_t = torch.tensor([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]).to(device_2)
    criterion_ce = FocalLoss(alpha_t,2)
    criterion_da = da_loss()
    alpha = 0.8
    beta = 0.2
    
    activation_EO = {}
    def getActivation_EO(name):
        def hook(model, input, output):
            activation_EO[name] = output.detach()
        return hook

    activation_SAR = {}
    def getActivation_SAR(name):
        def hook(model, input, output):
            activation_SAR[name] = output.detach()
        return hook
    
    h1_EO = model_EO.features[8].register_forward_hook(getActivation_EO('8'))
    h1_SAR = model_SAR.features[8].register_forward_hook(getActivation_SAR('8'))

    model_EO.to(device_1)
    model_SAR.to(device_2)
    
    optim_EO = optim.Adam(model_EO.parameters(),lr=0.003)
    optim_SAR = optim.Adam(model_SAR.parameters(),lr=0.003)
    scheduler_EO = ReduceLROnPlateau(optim_EO, 'min', patience=7)
    scheduler_SAR = ReduceLROnPlateau(optim_SAR, 'min', patience=7)

    print("Starting training on EO and SAR data")
    print()
    for epoch in range(20):
        train_loss_EO = 0.0
        train_loss_SAR = 0.0
        correct_EO = 0.0
        correct_SAR = 0.0
        total_EO = 0.0
        total_SAR = 0.0
        for i, (data1,data2) in enumerate(tqdm(train_loader)):
            inputs_EO, labels_EO, inputs_SAR, labels_SAR = data1[0], data1[1], data2[0], data2[1]
            inputs_EO, labels_EO, inputs_SAR, labels_SAR = inputs_EO.to(device_1), labels_EO.to(device_2), inputs_SAR.to(device_2), labels_SAR.to(device_2)
                                                     
            outputs_EO = model_EO(inputs_EO).to(device_2)
            outputs_SAR = model_SAR(inputs_SAR).to(device_2)

            h1 = []
            h1.append(activation_EO['8'])
            h1.append(activation_SAR['8'])

            EO_inp_unlabeled, SAR_inp_unlabeled = next(iter(unlabeled_train_loader))
            EO_inp_unlabeled, SAR_inp_unlabeled = EO_inp_unlabeled.to(device_1), SAR_inp_unlabeled.to(device_2)

            outputs_EO_unlabeled = model_EO(EO_inp_unlabeled)
            outputs_SAR_unlabeled = model_SAR(SAR_inp_unlabeled)

            h1.append(activation_EO['8'])
            h1.append(activation_SAR['8'])

            loss_ce_EO = criterion_ce(outputs_EO, labels_EO).to(device_1)
            loss_da = criterion_da(h1[0].to(device_2),h1[1].to(device_2)).to(device_1)
            loss_da_unlabeled = criterion_da(h1[2].to(device_2),h1[3].to(device_2)).to(device_1)
            loss_ce_SAR = criterion_ce(outputs_SAR, labels_SAR).to(device_1)
            loss_EO = loss_ce_EO+loss_ce_SAR+((alpha*loss_da)+(beta*loss_da_unlabeled))
            loss_SAR = loss_ce_SAR+loss_ce_EO+((alpha*loss_da)+(beta*loss_da_unlabeled))
            ################################################################################################################################################
            optim_EO.zero_grad()
            loss_EO.backward(retain_graph=True,inputs=list(model_EO.parameters()))
            optim_EO.step()
            optim_SAR.zero_grad()
            loss_SAR.backward(inputs=list(model_SAR.parameters()))
            optim_SAR.step()

            predictions_EO = outputs_EO.argmax(dim=1, keepdim=True).squeeze()
            correct_EO += (predictions_EO == labels_EO).sum().item()
            total_EO += labels_EO.size(0)
            
            predictions_SAR = outputs_SAR.argmax(dim=1, keepdim=True).squeeze()
            correct_SAR += (predictions_SAR == labels_SAR).sum().item()
            total_SAR += labels_SAR.size(0)
            
            train_loss_EO +=loss_EO.item()
            train_loss_SAR +=loss_SAR.item()

        accuracy_EO = correct_EO /total_EO
        accuracy_SAR = correct_SAR /total_SAR
        scheduler_EO.step(train_loss_EO)
        scheduler_SAR.step(train_loss_SAR)
        
        print('Loss_EO after epoch {:} is {:.2f} and accuracy_EO is {:.2f}'.format(epoch,(train_loss_EO / len(train_loader)),100.0*accuracy_EO))
        print('Loss_SAR after epoch {:} is {:.2f} and accuracy_SAR is {:.2f}'.format(epoch,(train_loss_SAR / len(train_loader)),100.0*accuracy_SAR))
        print()
        h1_EO.remove()
        h1_SAR.remove()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_SAR.state_dict(),
            'model' : model_SAR,
            'optimizer_state_dict': optim_SAR.state_dict()
            }, 'SAR_cross_domain_EffB0.pth')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_EO.state_dict(),
            'model' : model_EO,
            'optimizer_state_dict': optim_EO.state_dict()
            }, 'EO_cross_domain_EffB0.pth')
        
    print('Finished Simultaneous Training')
    print()
    torch.save(model_EO, 'EO_cross_domain_EffB0.pth')
    torch.save(model_SAR, 'SAR_cross_domain_EffB0.pth')
    print()

if __name__ == "__main__":
    train()
