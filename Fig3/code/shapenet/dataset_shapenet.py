import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
import pandas as pd

# adapted from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class

chosen_cats = {"airplane":0, "bed":1, "bottle":2, 
                "camera":3, "car":4, "chair":5}

class CustomDataset(Dataset):

    def __init__(self, csv_file, root_dir, train, computing_mean, normalize=None):

        self.img_names = pd.read_csv(f'../../data/shapenet/{csv_file}')
        self.train = train
        self.root_dir = root_dir
        self.computing_mean = computing_mean
        self.lt = ['t', 'tA', 'tB', 'tC', 'tD', 'tE', 'tF']
       
        self.all_transforms =  v2.Compose([
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(224, 224)),
        ])
        
        if normalize is not None:
            assert computing_mean == False
            self.top_transforms = v2.Compose([
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(224, 224)),
                v2.RandomHorizontalFlip(p=0.25),
                v2.RandomVerticalFlip(p=0.25),
                v2.RandomRotation(degrees=30),
                normalize,
            ])
            self.test_transforms = v2.Compose([
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(224, 224)),
                normalize,
            ])
        else:
            assert computing_mean == True
            self.top_transforms = v2.Compose([
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(224, 224)),
                v2.RandomHorizontalFlip(p=0.25),
                v2.RandomVerticalFlip(p=0.25),
                v2.RandomRotation(degrees=30),
            ])
            self.test_transforms = v2.Compose([
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True), 
                v2.Resize(size=(224, 224)),
            ])
        

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        obj_category = self.img_names.iloc[idx, 1]

        if self.train:
            suffix = "top" if (self.img_names.iloc[idx, 2] in self.lt) else "other"
            subfold = "shapenet_auto_images"

            my_dir = os.path.join(self.root_dir, subfold, obj_category, "frames", suffix)

        else:
            subfold = "shapeNet_custom_random"
            my_dir = os.path.join(self.root_dir, subfold, obj_category, "frames", "val")

        img_name = os.path.join(my_dir, self.img_names.iloc[idx, 0])
        img = Image.open(img_name)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        sample = {}

        if self.computing_mean:
            sample['img'] = self.all_transforms(img)
        else:
            if (self.img_names.iloc[idx, 2][0] == 't') and (self.train):
                sample['img'] = self.top_transforms(img)
            elif (self.img_names.iloc[idx, 2][0] == 'o') and (self.train):
                sample['img'] = self.top_transforms(img)
            elif (not self.train):
                sample['img'] = self.test_transforms(img)
            else:
                raise Exception("should not be here")

        sample['obj_cat'] = torch.tensor(chosen_cats[obj_category], dtype=torch.long)
        return sample