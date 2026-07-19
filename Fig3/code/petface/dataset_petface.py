import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
import pandas as pd

# adapted from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class

chosen_cats = {'javasparrow': 0, 'cat': 1, 'pig': 2, 'chimp': 3, 'dog': 4, 'ferret': 5}

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, train, computing_mean, size=None, normalize=None):

        self.img_names = pd.read_csv(f'../../data/petface/{csv_file}')
        assert self.img_names['category'].nunique() == 6

        self.train = train
        self.root_dir = root_dir
        self.computing_mean = computing_mean


        if size is not None:
            raise NotImplementedError()
       
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
                # v2.RandomHorizontalFlip(p=0.25),
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
                # v2.RandomHorizontalFlip(p=0.25),
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
        
        if self.train:
            col_id = 2
        else:
            col_id = 1
        obj_category = self.img_names.iloc[idx, col_id]
        img_name = os.path.join(self.root_dir, self.img_names.iloc[idx, 0])
        img = Image.open(img_name)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        sample = {}

        if self.computing_mean:
            sample['img'] = self.all_transforms(img)
        else:
            if self.train:
                sample['img'] = self.top_transforms(img)
            else:
                sample['img'] = self.test_transforms(img)

        sample['obj_cat'] = torch.tensor(chosen_cats[obj_category], dtype=torch.long)
        return sample