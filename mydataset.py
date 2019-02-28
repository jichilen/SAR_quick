from PIL import Image
from torch.utils.data import Dataset
import tqdm
import torch 

class MyDataset(Dataset):
    def __init__(self, txt_path,im_dir, transform = None, target_transform = None):
        with open(txt_path) as f:
            gts=f.readlines()
        imgs = []
        for gt in tqdm.tqdm(gts):
            imn=im_dir+gt.strip().split(' ')[0]
            la=gt.strip().split(' ')[1:]
            lat=torch.zeros(30)
            if len(la)>30:
                    continue
            for i in range(len(la)):
                lat[i]=int(la[i])
            imgs.append([imn,lat])
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
