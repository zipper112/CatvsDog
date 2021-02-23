import os 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random
import config
import matplotlib.pyplot as plt

class Cat_Vs_DogDatasets(Dataset):
    def __init__(self, path: str, imglist: list, s: int=config.PIC_SIZE):
        self.path = path
        self.compose = transforms.Compose([
            transforms.Resize(size=s),
            transforms.CenterCrop(size=s),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.imglist = imglist
        self.len = len(imglist)
    
    def __getitem__(self, idx: int):
        name = self.imglist[idx]
        lable = 0
        if name[:3] == 'cat':
            lable = 1
        return self.compose(Image.open(os.path.join(self.path, name) ,mode='r')), torch.tensor(lable, dtype=torch.int64)
    
    def __len__(self):
        return self.len




def train_test_split(path, test_size=0.15, random_state:int=666, s: int=config.PIC_SIZE):
    imglist = os.listdir(path)
    random.seed(random_state)
    random.shuffle(imglist)
    train_size = int((1 - test_size) * len(imglist))
    train = imglist[:train_size]
    test = imglist[train_size:]
    return Cat_Vs_DogDatasets(path, imglist=train, s=s), Cat_Vs_DogDatasets(path, imglist=test, s=s)
