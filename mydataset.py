from PIL import Image
from torch.utils.data import Dataset
import tqdm
import torch
import os

class MyDataset(Dataset):

    def __init__(self, im_dir, txt_path=None, transform=None, target_transform=None):
        if txt_path:
            with open(txt_path) as f:
                gts = f.readlines()
            imgs = []
            for gt in tqdm.tqdm(gts):
                imn = im_dir + gt.strip().split(' ')[0]
                la = gt.strip().split(' ')[1:]
                if len(la) > 30:
                    continue
                
                imgs.append([imn, la])
        else:
            imgs=[]
            ims=os.listdir(im_dir)
            for im in ims:
                imgs.append([os.path.join(im_dir,im),torch.zeros(30)])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        try:
            img = Image.open(fn).convert('RGB')
        except IOError:
            print('Corrupted image for %s' % fn)
            return self[index+1]
        lat = torch.zeros(30)
        for i in range(len(label)):
            lat[i] = int(label[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, lat

    def __len__(self):
        return len(self.imgs)


class My90kDataset(Dataset):

    def __init__(self, im_dir, txt_path=None, transform=None, target_transform=None):
        if txt_path:
            with open(txt_path) as f:
                gts = f.readlines()
            imgs = []
            for gt in tqdm.tqdm(gts):
                imn = im_dir + gt.strip().split(' ')[0]
                la = gt.strip().split(' ')[1]
                la=[ord(x)-ord('a')+1 for x in la]
                lat = torch.zeros(30)
                if len(la) > 30:
                    continue
                for i in range(len(la)):
                    lat[i] = int(la[i])
                imgs.append([imn, lat])
        else:
            imgs=[]
            ims=os.listdir(im_dir)
            for im in ims:
                imgs.append([os.path.join(im_dir,im),torch.zeros(30)])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)