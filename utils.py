from PIL import Image
import numpy as np
import cv2
import tqdm
import os
import sys

import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import BatchSampler

def get_char_dict():
    chardict=[]
    for i in range(1,37):
        if i <11:
            chardict.append(chr(ord('0')+i-1))
        else:
            chardict.append(chr(ord('a')+i-11))
    return chardict
 
class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations

def prepare_data(save_dir, data_name, is_training):
    '''
    load data and preprocess data

    '''
    data_path = {
        ''
    }
    postfix = '.pth'
    assert len(data_name)>0,'data_name is empty'
    if not isinstance(data_name, list):
        data_name = [data_name]
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((48, 160)),  #(h, w)
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    # Lists can be seen as the simplest class of dataset, which can not hold
    # any other information.
    datasets = []
    for data_n in data_name:
        dataset = []
        print(os.path.join(save_dir, data_n + postfix))
        if os.path.exists(os.path.join(save_dir, data_n + postfix)):
            dataset = torch.load(os.path.join(save_dir, data_n + postfix))
        else:
            # only icpr now
            imdir='./data/icpr/crop/'
            with open('./data/icpr/char2num.txt') as f:
                gts=f.readlines()
            cal=0
            for gt in tqdm.tqdm(gts):
                imn=gt.strip().split(' ')[0]
                la=gt.strip().split(' ')[1:]
                lat=torch.zeros(30)
                if len(la)>30:
                    continue
                cal+=1
                if cal>1000:
                    break
                for i in range(len(la)):
                    lat[i]=int(la[i])
                im=Image.open(imdir+imn).convert('RGB')
                im=transform(im)
                dataset.append([im,lat])
        torch.save(dataset,save_dir+'/icpr.pth')
        datasets.extend(dataset)
    return datasets