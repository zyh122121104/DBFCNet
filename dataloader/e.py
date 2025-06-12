import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms as transforms
from PIL import Image,ImageEnhance,ImageOps
from torchvision.io import read_video, write_video


class CustomDataset(Dataset):
    def __init__(self, data,transforms=None):
        self.data = data  # 读取Excel文件
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient = self.data.loc[idx,'ID']
        label = self.data.loc[idx, 'label']

        type1_paths = self.data.loc[idx, 'ceus_path']
        type2_paths = self.data.loc[idx, 'bmodel_path']
        type3_paths = self.data.loc[idx, 'seg_path']

        # 读取并合并图片为视频
        ceus_video = self.images_to_video(type1_paths)

        bmodel = self.images(type2_paths)

        seg = self.seg(type3_paths)

        return (seg,ceus_video,bmodel),label

        # return ceus_video, label

    def variance(self,image_path):
        img = Image.open(image_path).convert('L')
        image_array = np.array(img)
        image_max = np.max(image_array)
        variance = np.var(image_array)/image_max

        return variance


    def variance_to_images(self,image_paths):
        image_path_first = image_paths[0].replace("'", "")
        image_path_last = image_paths[-3].replace("'", "")
        return self.variance(image_path_last) - self.variance(image_path_first)

    def images_to_video(self, image_paths):
        # 读取所有图片并组成视频
        frames = []
        image_paths = image_paths.strip('[]')
        image_paths = image_paths.split(', ')
        if self.variance_to_images(image_paths) < 2.0:
            numbers1 = np.linspace(0, 3, 14)
            numbers2 = np.linspace(3,0,2)
            numbers = np.concatenate((numbers1,numbers2))
            numbers = 2 + (4 - 2) * (np.exp(numbers) - 1) / (np.exp(3) - 1)
            for i,img_path in enumerate(image_paths):
                img_path = img_path.replace("'", "")
                img = Image.open(img_path)
                # img = ImageEnhance.Contrast(img)
                # factor = numbers[i]
                # img = img.enhance(factor)
                if self.transforms:
                    img = self.transforms(img)
                frames.append(img)
        else:
            for img_path in image_paths:
                img_path = img_path.replace("'", "")
                img = Image.open(img_path)
                if self.transforms:
                    img = self.transforms(img)
                frames.append(img)

        # 检查frames是否为空
        if not frames:
            return torch.empty(0)

        # 将frames转换为视频
        video_tensor = torch.stack(frames).permute(1, 0, 2, 3)
        return video_tensor

    def images(self, image_paths):
        # 读取所有图片并组成视频
        frames = []
        image_paths = image_paths.strip('[]')
        image_paths = image_paths.split(', ')
        img_path = image_paths[-1].replace("'", "")
        # img = Image.open(img_path).convert('L')
        img = Image.open(img_path)
        # img = ImageOps.equalize((img))
        if self.transforms:
            img = self.transforms(img)

        return img

    def seg(self, image_paths):
        # 读取所有图片并组成视频
        frames = []
        image_paths = image_paths.strip('[]')
        image_paths = image_paths.split(', ')
        img_path = image_paths[0].replace("'", "")
        img = Image.open(img_path).convert('L')
        if self.transforms:
            img = self.transforms(img)
        return img


class RandomTransform(object):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __call__(self,frame):
        # if random.random() < 0.5:
        #     frame = frame.flip(1)

        angle = random.randint(-20,20)
        frame = transforms.functional.rotate(frame,angle)

        return self.transform(frame)

class Transform(object):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __call__(self,frame):
        # if random.random() < 0.5:
        #     frame = frame.flip(1)

        # angle = random.randint(-20,20)
        # frame = transforms.functional.rotate(frame,angle)

        return self.transform(frame)

def create_dataloader(excel_file,transform):
    # 定义视频数据的转换操作，可以根据需要添加其它变换
    # data = pd.read_excel(excel_file)
    dataset = CustomDataset(excel_file,transforms=transform)
    return dataset

# # 使用示例
# excel_path = '/home/ubuntu/Documents/cgl/CEUS_CLASSIFICATION/dataloader/select2.xlsx'
# df = pd.read_excel(excel_path)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # 调整大小
#     transforms.ToTensor(),          # 转为张量
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
# ])
# dataset = create_dataloader(df,transform=transform)
# dataload = DataLoader(dataset,batch_size=1,num_workers=16)
# i=0
# patient = []
# l = []
# for idx,(p,image,label) in enumerate(dataload):
#     # if torch.max(image) > 1.0:
#     #     print(torch.max(image))
#     #     print(p)
#     if torch.var(image) < 0.07:
#         patient.append(p[0])
#         l.append(label.item())
#         print(p[0])
#         print(label)
#         i+=1
#     continue
# s_ex = pd.DataFrame({
#     'patient': patient,
#     'label': l
# })
# s_ex.to_excel(f"/home/ubuntu/Documents/cgl/CEUS_CLASSIFICATION/excel/excel_{0.07}.xlsx",index=False,engine='openpyxl')
# print(i)


