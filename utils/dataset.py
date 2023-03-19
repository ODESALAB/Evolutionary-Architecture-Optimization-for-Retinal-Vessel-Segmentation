import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip
from utils.helpers import Fix_RandomRotation

#====================================================================
# This part of the code is based on FR-UNet
# from https://github.com/lseventeen/FR-UNet
# Liu, W., Yang, H., Tian, T., Cao, Z., Pan, X., Xu, W., ... & Gao, F. (2022). Full-resolution network and dual-threshold iteration for retinal vessel and coronary angiograph segmentation. IEEE Journal of Biomedical and Health Informatics, 26(9), 4623-4634.
# ===================================================================

class vessel_dataset(Dataset):
    def __init__(self, path, mode, is_val=False, split=None, de_train=False):

        self.mode = mode
        self.is_val = is_val
        self.de_train = de_train
        self.data_path = os.path.join(path, f"{mode}_pro")
        self.data_file = os.listdir(self.data_path) # CHASEDB1
        
        """
        # DRIVE
        self.data_file = None
        if mode == 'test':
            self.data_file = os.listdir(self.data_path)
        else:
            self.data_file = self.readIndexes(os.path.join(path, "short_run.txt"))
        """
        self.img_file = self._select_img(self.data_file)
        if split is not None and mode == "training":
            assert split > 0 and split < 1
            if not is_val:
                self.img_file = self.img_file[:int(split*len(self.img_file))]
            else:
                self.img_file = self.img_file[int(split*len(self.img_file)):]
        self.transforms = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])

    def __getitem__(self, idx):
        img_file = self.img_file[idx]
        with open(file=os.path.join(self.data_path, img_file), mode='rb') as file:
            img = torch.from_numpy(pickle.load(file)).float()
        gt_file = "gt" + img_file[3:]
        with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
            gt = torch.from_numpy(pickle.load(file)).float()

        if self.mode == "training" and not self.is_val:
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transforms(img)
            torch.manual_seed(seed)
            gt = self.transforms(gt)

        return img, gt

    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                img_list.append(file)

        return img_list

    def __len__(self):
        if self.de_train == False:
            return len(self.img_file)
        else:
            return len(self.img_file) // 2

    def readIndexes(self, path):
        lines = []
        with open(f"{path}", "r") as f:
            lines = f.readlines()
        
        lines = [fname.replace("\n","") for fname in lines]
        return lines
