from PIL import Image
from torch.utils.data import Dataset
import tqdm
import torch
import os
import scipy.io as sio
import random

def tol(l):
    l = l.lower()
    if ord(l) < ord('a'):
        return ord(l) - ord('0') + 1
    else:
        return ord(l) - ord('a') + 11


class MyDataset(Dataset):

    def __init__(self, datanames, transform=None, target_transform=None):
        if not isinstance(datanames,list):
            datanames=[datanames]
        imgs=[]
        for dataname in datanames:
            im_dir, txt_path = self.get_dataset(dataname)
            imgs.extend(self.get_data(im_dir, txt_path))
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
        if img.height>img.width:
            if random.random()>0.5:
                img=img.transpose(Image.ROTATE_90)
            else:
                img=img.transpose(Image.ROTATE_270)
        lat = torch.zeros(30)
        for i in range(len(label)):
            lat[i] = int(label[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, lat

    def __len__(self):
        return len(self.imgs)

    def get_dataset(self, dataname):
        dataset = {
            '90kDICT32px_train': ['/data2/data/90kDICT32px/', '/data2/data/90kDICT32px/Synth_train_sample.txt'],
            '90kDICT32px_val': ['/data2/data/90kDICT32px/', '/data2/data/90kDICT32px/Synth_val_test.txt'],
            'IIIT5K_train': ['/data2/text/recognition/IIIT5K/', '/data2/text/recognition/IIIT5K/trainCharBound.mat'],
            'IIIT5K_test': ['/data2/text/recognition/IIIT5K/', '/data2/text/recognition/IIIT5K/testCharBound.mat'],
            'SynthChinese_train': ['/data2/text/recognition/SynthChinese/images/', '/data2/text/recognition/SynthChinese/train.txt'],
            'SynthChinese_test': ['/data2/text/recognition/SynthChinese/images/', '/data2/text/recognition/SynthChinese/test_test.txt'],
            'icpr': ['/data2/text/recognition/recognition/icpr/crop/', '/data2/text/recognition/recognition/icpr/char2num.txt'],
            'expr0': ['/data2/text/recognition/recognition/expr0/crop/', '/data2/text/recognition/recognition/expr0/char2num.txt'],
            'test': ['/data2/text/recognition/recognition/imgs2_east_regions/', '/data2/text/recognition/recognition/test.txt'],
        }
        return dataset[dataname]

    def get_data(self, im_dir, txt_path=None):
        if txt_path:
            if 'mat' in txt_path:
                data = sio.loadmat(txt_path)
                da = data[txt_path.split('/')[-1][:-4]][0]
                imgs = []
                for gt in tqdm.tqdm(da, ascii=True):
                    imn = im_dir + gt[0][0]
                    la = gt[1][0].strip()
                    if len(la) > 30:
                        continue
                    la = [tol(l) for l in la]
                    imgs.append([imn, la])
            else:
                with open(txt_path) as f:
                    gts = f.readlines()
                imgs = []
                for gt in tqdm.tqdm(gts, ascii=True):
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
        return imgs


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
