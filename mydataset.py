from PIL import Image
from torch.utils.data import Dataset
import tqdm
import torch
import os
import scipy.io as sio

class MyDataset(Dataset):

    def __init__(self, dataname, transform=None, target_transform=None):

        im_dir, txt_path = self.get_dataset(dataname)
        if txt_path:
            if 'mat' in txt_path:
                data = sio.loadmat(txt_path)
                da=data['trainCharBound'][0]
                imgs=[]
                for gt in tqdm.tqdm(da):
                    imn = im_dir + gt[0][0]
                    la=gt[1][0].strip()
                    if len(la) > 30:
                        continue
                    imgs.append([imn, la])
            else:
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
            imgs = []
            ims = os.listdir(im_dir)
            for im in ims:
                imgs.append([os.path.join(im_dir, im), [0]])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        try:
            img = Image.open(fn).convert('RGB')
        except IOError:
            print('Corrupted image for %s' % fn)
            return self[index + 1]
        lat = torch.zeros(30)
        for i in range(len(label)):
            lat[i] = int(label[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, lat

    def __len__(self):
        return len(self.imgs)

    def get_dataset(self,dataname):
        dataset = {
            '90kDICT32px_train': ['/data2/data/90kDICT32px/','/data2/data/90kDICT32px/Synth_train_spilt.txt'],
            '90kDICT32px_val': ['/data2/data/90kDICT32px/','/data2/data/90kDICT32px/Synth_val_test.txt'],
            'IIIT5K_train': ['/data2/text/character_detection/IIIT5K/train/','/data2/text/character_detection/IIIT5K/trainCharBound.mat'],
            'IIIT5K_test': ['/data2/text/character_detection/IIIT5K/test/','/data2/text/character_detection/IIIT5K/testCharBound.mat'],
        }
        return dataset[dataname]


class My90kDataset(Dataset):

    def __init__(self, im_dir, txt_path=None, transform=None, target_transform=None):
        if txt_path:
            with open(txt_path) as f:
                gts = f.readlines()
            imgs = []
            for gt in tqdm.tqdm(gts):
                imn = im_dir + gt.strip().split(' ')[0]
                la = gt.strip().split(' ')[1]
                la = [ord(x) - ord('a') + 1 for x in la]
                lat = torch.zeros(30)
                if len(la) > 30:
                    continue
                for i in range(len(la)):
                    lat[i] = int(la[i])
                imgs.append([imn, lat])
        else:
            imgs = []
            ims = os.listdir(im_dir)
            for im in ims:
                imgs.append([os.path.join(im_dir, im), torch.zeros(30)])
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
