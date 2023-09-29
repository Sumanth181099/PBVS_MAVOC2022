import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import cv2

device_ids=[0]
device = f'cuda:{device_ids[0]}'
class InfDataset(Dataset):
    def __init__(self, img_folder1, img_folder2, transform=None):
        self.imgs_folder1 = img_folder1
        self.imgs_folder2 = img_folder2
        self.transform = transform
        self.img_paths1 = []
        self.img_paths2 = []

        img_path1 = self.imgs_folder1 + '/'
        img_list1 = os.listdir(img_path1)
        img_list1.sort()

        img_path2 = self.imgs_folder2 + '/'
        img_list2 = os.listdir(img_path2)
        img_list2.sort()

        self.img_nums1 = len(img_list1)
        self.img_nums2 = len(img_list2)

        for i in range(self.img_nums1):
            img_name1 = img_path1 + img_list1[i]
            self.img_paths1.append(img_name1)
            img_name2 = img_path2 + img_list2[i]
            self.img_paths2.append(img_name2)

    def __getitem__(self, idx):
        img1 = cv2.imread(self.img_paths1[idx])
        img2 = cv2.imread(self.img_paths2[idx])
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        name1 = self.img_paths1[idx]
        name1 = os.path.basename(name1)
        name1 = name1.split("_")[1]
        name1 = name1.split(".")[0]

        name2 = self.img_paths2[idx]
        name2 = os.path.basename(name2)
        name2 = name2.split("_")[1]
        name2 = name2.split(".")[0]
        return (img1,name1), (img2,name2)
        
    def __len__(self):
        return self.img_nums1

inf_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#################### Please input the path of the EO and SAR test images in img_folder1 and img_folder2 respectively ##################################################
img_folder1='/home/aniruddh/Desktop/PBVS_Exec_Submission/Track2(SAR+EO)/data/NTIRE2021_Class_test_images_EO'
img_folder2='/home/aniruddh/Desktop/PBVS_Exec_Submission/Track2(SAR+EO)/data/NTIRE2021_Class_test_images_SAR'
inf_dataset = InfDataset(img_folder1, img_folder2, transform=inf_transform)
inf_dataloader = data.DataLoader(inf_dataset, batch_size=64, shuffle=True)

def test():
    model_EO_Resnet = torch.load('EO_cross_domain_resnet50.pth')
    model_EO_Resnet.to(device)
    image_id_list=[]
    class_id_list=[]
    model_EO_Resnet.eval()
    with torch.no_grad():
        for batch_idx, (data1, data2) in tqdm(enumerate(inf_dataloader)):
            img_EO, name_EO, img_SAR, name_SAR = data1[0],data1[1], data2[0],data2[1]
            img_EO, img_SAR = img_EO.to(device), img_SAR.to(device)
            output_unlabeled_EO_Resnet = model_EO_Resnet(img_EO)
            output_unlabeled_SAR_Resnet = model_EO_Resnet(img_SAR)
            output_unlabeled = torch.add(0.8*output_unlabeled_EO_Resnet,0.2*output_unlabeled_SAR_Resnet)
            _, pseudo_labeled = torch.max(output_unlabeled, 1)
############## Please have an empty results.csv file ready  in the working directory with the headers as 'image_id' and 'class_id' so that the outputs can be appended to the csv file #################
            for i in range(len(name_EO)):
                image_id_list.append(int(name_EO[i]))
                image_id_list.append(int(name_SAR[i]))
                class_id_list.append(pseudo_labeled[i].cpu().numpy())
                class_id_list.append(pseudo_labeled[i].cpu().numpy())
            
    df = pd.DataFrame({'image_id':image_id_list,
                       'class_id':class_id_list})
    
    df.to_csv('results.csv', mode='a', index=False, header=False)

test()
