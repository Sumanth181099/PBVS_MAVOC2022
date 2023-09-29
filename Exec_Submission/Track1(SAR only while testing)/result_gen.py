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
    def __init__(self, img_folder, transform=None):
        self.imgs_folder = img_folder
        self.transform = transform
        self.img_paths = []

        img_path = self.imgs_folder + '/'
        img_list = os.listdir(img_path)
        img_list.sort()

        self.img_nums = len(img_list)

        for i in range(self.img_nums):
            img_name = img_path + img_list[i]
            self.img_paths.append(img_name)
            
    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        if self.transform:
            img = self.transform(img)

        name = self.img_paths[idx]
        name = os.path.basename(name)
        name = name.split("_")[1]
        name = name.split(".")[0]
        return (img,name)

    def __len__(self):
        return self.img_nums

inf_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#################### Please input the path of the SAR test images in img_folder ##################################################
img_folder='/home/aniruddh/Desktop/PBVS_Exec_Submission/Track1(SAR)/data/NTIRE2021_Class_test_images_SAR'
inf_dataset = InfDataset(img_folder, transform=inf_transform)
inf_dataloader = data.DataLoader(inf_dataset, batch_size=64, shuffle=True)

def test():
    model_SAR_Resnet = torch.load('SAR_cross_domain_resnet50_testdata.pth')
    model_SAR_Eff = torch.load('SAR_cross_domain_EffB0_testdata.pth')

    model_SAR_Resnet.to(device)
    model_SAR_Eff.to(device)

    image_id_list=[]
    class_id_list=[]
    model_SAR_Resnet.eval()
    model_SAR_Eff.eval()

    with torch.no_grad():
        for batch_idx, (img, name) in tqdm(enumerate(inf_dataloader)):
            img=img.to(device)
            output_unlabeled_SAR_Resnet = model_SAR_Resnet(img)
            output_unlabeled_SAR_Eff = model_SAR_Eff(img)
            output_unlabeled_SAR = torch.add(0.7*output_unlabeled_SAR_Resnet,0.3*output_unlabeled_SAR_Eff)
            _, pseudo_labeled = torch.max(output_unlabeled_SAR, 1)
############## Please have an empty results.csv file ready  in the working directory with the headers as 'image_id' and 'class_id' so that the outputs can be appended to the csv file #################
            for i in range(len(name)):
                image_id_list.append(int(name[i]))
                class_id_list.append(pseudo_labeled[i].cpu().numpy())

    df = pd.DataFrame({'image_id':image_id_list,
                       'class_id':class_id_list})

    df.to_csv('results.csv', mode='a', index=False, header=False)

test()
