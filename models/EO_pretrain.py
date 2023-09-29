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

#writer = SummaryWriter('runs/da_valid_file_train_tail_1')
torch.autograd.set_detect_anomaly(True)
device_ids=[1]
device = f'cuda:{device_ids[0]}'
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dataset_EO(Dataset):
    def __init__(self, datasetA):
        self.datasetA = datasetA

    def __getitem__(self, index):
        (xA,lA) = (self.datasetA[index][0],self.datasetA[index][1])
        return (xA,lA)
    
    def __len__(self):
        return len(self.datasetA)

transform = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
EO_file_pth = '/home/airl-gpu4/Aniruddh/PBVS Challenge/Datasets/train_images/train_images/EO_data'

train_data_EO = torchvision.datasets.ImageFolder(root=EO_file_pth,transform=transform)

train_dataset_size_EO = train_data_EO.__len__()
train_dataset = Dataset_EO(train_data_EO)
################################ weighted random sampler for training for EO domain ####################################################
y_train_indices_EO = len(train_data_EO)

y_train_EO = [train_data_EO.targets[i] for i in range(y_train_indices_EO)]

class_sample_count = np.array(
    [len(np.where(y_train_EO == t)[0]) for t in np.unique(y_train_EO)])

weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in y_train_EO])
samples_weight = torch.from_numpy(samples_weight)

sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),replacement=False)
#train_loader = data.DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=5)
train_loader = data.DataLoader(train_dataset, batch_size=64, num_workers=5)
#####################################################################################################################################################
def train():
    model_EO = models.resnet50(pretrained=True)
    num_ftrs_EO = model_EO.fc.in_features
    model_EO.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs_EO, 10))

    for param in model_EO.parameters():
        param.requires_grad = True

    model_EO.to(device)
    alpha_t = torch.tensor([0.125,1.04,1.92,2.76,16.87,34.48,35.48,47.08,34.97,46.41]).to(device)
    criterion = FocalLoss(alpha_t,2)
    
    optim_EO = optim.Adam(model_EO.parameters(),lr=0.003)
    scheduler_EO = ReduceLROnPlateau(optim_EO, 'min', patience=7)

    print("Starting training on EO data")
    print()
    for epoch in range(15):
        train_loss_EO = 0.0
        correct_EO = 0.0
        total_EO = 0.0
        for i, data in enumerate(train_loader):
            inputs_EO, labels_EO = data[0], data[1]
            inputs_EO, labels_EO = inputs_EO.to(device), labels_EO.to(device)

            outputs_EO = model_EO(inputs_EO)

            loss_EO = criterion(outputs_EO,labels_EO)
            
            optim_EO.zero_grad()
            loss_EO.backward()
            optim_EO.step()
            
            predictions_EO = outputs_EO.argmax(dim=1, keepdim=True).squeeze()
            correct_EO += (predictions_EO == labels_EO).sum().item()
            total_EO += labels_EO.size(0)
            
            train_loss_EO +=loss_EO.item()

        accuracy_EO = correct_EO /total_EO
        scheduler_EO.step(train_loss_EO)
        print('Loss_EO after epoch {:} is {:.2f} and accuracy_EO is {:.2f}'.format(epoch,(train_loss_EO / len(train_loader)),100.0*accuracy_EO))
        print()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_EO.state_dict(),
            'model' : model_EO,
            'optimizer_state_dict': optim_EO.state_dict()
            }, 'EO_focal_loss_pretrained_resnet50.pth')
        
    print('Finished Pre-training')
    print()
    torch.save(model_EO, 'EO_focal_loss_pretrained_resnet50.pth')

if __name__ == "__main__":
    train()
