# from collections import OrderedDict
import os
import argparse
from PIL import Image
import numpy as np
import cv2
import tqdm
import time
import sys

import torch
from torch import nn
from torchsummary import summary
import torch.utils.data as data
import torchvision.transforms as transforms

import resnet
import recog
from utils import prepare_data
from utils import Logger
# from utils import IterationBasedBatchSampler
from mydataset import MyDataset


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.conv_layer = resnet.ResNet(31)
        self.recg = recog.Recogniton(args)

    def forward(self, image, target=None):
        conv_f = self.conv_layer(image)
        if self.training:
            out = self.recg(conv_f, target)
        else:
            seq, seq_score = self.recg(conv_f)  # 2 32 | 2 32
            return seq, seq_score
        return out


def train(args, local_rank, distributed, trainset):
    model = Model(args)
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    params = []
    for key, value in model.named_parameters():
        # print(key)
        if not value.requires_grad:
            continue
        lr = args.lr # cfg.SOLVER.BASE_LR
        weight_decay = 0.0001  # cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = args.lr * 2  # cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = 0  # cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # cfg.SOLVER.MOMENTUM
    # optimizer = torch.optim.SGD(params, lr, momentum=0.9)
    optimizer = torch.optim.Adam(params, lr)
    model.train()
    # data loader
    
    sampler = torch.utils.data.sampler.RandomSampler(trainset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, args.batch_size, drop_last=True
        )
    trainloader = data.DataLoader(
        trainset, batch_sampler=batch_sampler, num_workers=2)
    # trainloader = data.DataLoader(
    #     trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    total_loss = 0
    dis_val = 10
    epoc_t = 201
    save_epoc=1
    num_b = len(trainloader)
    outstr = "epoc: {:3d}({}), iter: {:4d}({}), loss: {:.4f}, ave_l: {:.4f}, time: {:.4f}"
    start_training_time = time.time()
    for epoc in range(epoc_t):
        for batch_idx, (imgs, targets) in enumerate(trainloader):
            imgs = imgs.cuda()
            targets = targets.cuda()
            losses = model(imgs, targets)
            # cal loss
            optimizer.zero_grad()
            losses.backward(retain_graph=True)
            optimizer.step()
            total_loss += losses.item()
            if batch_idx % dis_val == 0:
                print(outstr.format(epoc, epoc_t, batch_idx, num_b, losses.item(
                ), total_loss / dis_val, time.time() - start_training_time))
                total_loss = 0
                start_training_time = time.time()
        if epoc > 0 and epoc % save_epoc == 0:
            state = {'net': model.state_dict(
            ), 'optimizer': optimizer.state_dict(), 'epoc': epoc}
            torch.save(state, './model/save_e_' + str(epoc) + '.th')
            if int(epoc / save_epoc)>3:
                if os.path.exists('./model/save_e_' + str(epoc-save_epoc*3) + '.th'):
                    os.remove('./model/save_e_' + str(epoc-save_epoc*3) + '.th')


def test(args, local_rank, distributed, trainset):
    args.batch_size = 1
    print(args)
    model = Model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    state = torch.load('./model/save_e_150.th')
    model.load_state_dict(state['net'])
    model.eval()
    # data loader
    trainloader = data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=2)
    for batch_idx, (imgs, targets) in enumerate(trainloader):
        imgs = imgs.cuda()
        targets = targets.cuda()
        seq, scores = model(imgs, targets)
        print(seq)
        print(scores)
        print(targets)

    ''' test code
    img = torch.rand(2, 3, 48, 160).cuda()
    target = torch.zeros(2, 30).cuda()
    target[:, :10] = 1
    target[:, 10:19] = 2
    '''

    # model.eval()
    # seq, seq_score = model(img, target)
    # print(seq, seq_score)


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--save_dir", default="./data",
                        help="path to save dataset", type=str,)
    parser.add_argument(
        "-d", "--data_name", default=['icpr'], help="path to save dataset", type=str, nargs='+',)
    parser.add_argument("--is_training", default=True,
                        help="training or evaluation", type=bool,)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    args.batch_size = 24
    args.num_layers = 2
    args.featrue_layers = 512
    args.hidden_dim = 512
    args.vocab_size = 6498
    args.out_seq_len = 30
    args.hidden_dim_de = 512
    args.max_h = 6
    args.max_w = 40
    args.embedding_size = 512
    args.lr=0.02
    args.checkpoint_dir='./model/adam'

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    sys.stdout = Logger(args.checkpoint_dir+'/log.txt')

    num_gpus = int(os.environ["WORLD_SIZE"]
                   ) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.distributed = False  # not implement
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    if args.is_training:
        transform = transforms.Compose([
            transforms.Resize((48, 160)),  # (h, w)
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
    trainset = MyDataset('./data/icpr/char2num.txt',
                         './data/icpr/crop/', transform)
    # trainset = prepare_data(args.save_dir, args.data_name, args.is_training)
    if args.is_training:
        train(args, args.local_rank, args.distributed, trainset)
    else:
        test(args, args.local_rank, args.distributed, trainset)


if __name__ == '__main__':

    main()
    # model = Model()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # device = torch.device('cpu')
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    # model.to(device)
    # # model.eval()
    # model.train()
    # # print(model)
    # # summary(model, (3, 48, 160))  # [-1, 512, 6, 40] nchw
    # img = torch.Tensor(2, 3, 48, 160).cuda()
    # target = torch.zeros(2, 30).cuda()
    # target[:, :19] = 1
    # # out, out1 = model(img)
    # # print(out.shape, out1.shape)
    # out=model(img,target)
    # print(out)
