# from collections import OrderedDict
import os
import argparse
from PIL import Image
import numpy as np
import cv2
import tqdm
import time
import sys
import editdistance

import torch
from torch import nn
from torchsummary import summary
import torch.utils.data as data
import torchvision.transforms as transforms

import resnet
import recog
from utils import prepare_data, get_char_dict
from utils import Logger
# from utils import IterationBasedBatchSampler
from mydataset import MyDataset, My90kDataset


def model_class_merge(model, optimizer, state, new_size):
    new_size += 2
    # recg.embed.weight recg.linear_out.bias recg.linear_out.weight
    net_p = state['net']
    old_size = net_p['recg.embed.weight'].shape[0]
    if old_size == new_size:
        model.load_state_dict(net_p)
        optimizer.load_state_dict(state['optimizer'])
        return
    for k, v in net_p.items():
        if 'recg.embed.weight' in k or 'recg.linear_out.bias' in k or 'recg.linear_out.weight' in k:
            shape_n = [s1 for s1 in net_p[k].shape]
            shape_n[0] = new_size
            var = net_p[k].new_zeros(shape_n)
            var[:old_size, ...] = net_p[k]
            net_p[k] = var
    model.load_state_dict(net_p)


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
    # basic settings
    model = Model(args)
    print(args)
    model.to(args.device)
    params = []

    # fix the bias lr
    for key, value in model.named_parameters():
        # print(key)
        if not value.requires_grad:
            continue
        lr = args.lr  # cfg.SOLVER.BASE_LR
        weight_decay = 0.0001  # cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = args.lr * 2  # cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = 0  # cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # data optimizer
    # cfg.SOLVER.MOMENTUM
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(params, lr)
    else:
        optimizer = torch.optim.SGD(params, lr, momentum=0.9)
    bg_iter = 0

    # fintune from exist ckpt
    if os.path.exists(args.checkpoint_dir + '/last_checkpoint.txt'):
        with open(args.checkpoint_dir + '/last_checkpoint.txt') as f:
            lck = f.readline().strip()
        print("fintune from " + lck)
        state = torch.load(args.checkpoint_dir + lck)
        model_class_merge(model, optimizer, state, args.vocab_size)
        # model.load_state_dict(state['net'], strict=False)
        # optimizer.load_state_dict(state['optimizer'])
        bg_iter = state['batch_idx']
        bg_epoc = state['epoc']
    elif args.ckpt:
        print("fintune from " + args.ckpt)
        state = torch.load(args.ckpt)
        model.load_state_dict(state['net'], strict=False)
        
    model.train()

    # data loader
    sampler = torch.utils.data.sampler.RandomSampler(trainset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, args.batch_size, drop_last=False
    )
    trainloader = data.DataLoader(
        trainset, batch_sampler=batch_sampler, num_workers=2)
    # trainloader = data.DataLoader(
    #     trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # some basic settings
    total_loss = 0
    dis_val = 10
    epoc_t = 201
    save_epoc = 1
    save_batch = 500
    num_b = len(trainloader)
    outstr = "epoc: {:3d}({}), iter: {:4d}({}), loss: {:.4f}, ave_l: {:.4f}, time: {:.4f}, lr: {:.5f}"
    start_training_time = time.time()

    # begin loop
    for epoc in range(epoc_t):
        for batch_idx, (imgs, targets) in enumerate(trainloader, bg_iter):
            if batch_idx >= len(trainloader) - 1:
                bg_iter = 0
                break
            imgs = imgs.cuda()
            targets = targets.cuda()
            losses = model(imgs, targets)
            # cal loss
            optimizer.zero_grad()
            losses.backward(retain_graph=True)
            optimizer.step()
            total_loss += losses.item()
            if batch_idx % dis_val == 0:
                print(outstr.format(epoc, epoc_t, batch_idx, num_b, losses.item(), total_loss /
                                    dis_val, time.time() - start_training_time, optimizer.param_groups[0]["lr"]))
                total_loss = 0
                start_training_time = time.time()
            if batch_idx > 0 and batch_idx % save_batch == 0:
                state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(
                ), 'epoc': epoc, 'batch_idx': batch_idx}
                torch.save(state, args.checkpoint_dir +
                           str(epoc) + '-' + str(batch_idx) + '.th')
                with open(args.checkpoint_dir + 'last_checkpoint.txt', 'w') as f:
                    print('writing ' + str(epoc) + '-' +
                          str(batch_idx) + '.th to' + args.checkpoint_dir)
                    f.write(str(epoc) + '-' + str(batch_idx) + '.th')
                if int(batch_idx / save_batch) > 3:
                    if os.path.exists(args.checkpoint_dir + str(epoc) + '-' + str(batch_idx - save_batch * 3) + '.th'):
                        os.remove(args.checkpoint_dir + str(epoc) + '-' +
                                  str(batch_idx - save_batch * 3) + '.th')
        if epoc % save_epoc == 0:
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(
            ), 'epoc': epoc, 'batch_idx': batch_idx}
            torch.save(state, args.checkpoint_dir + str(epoc) + '-' +
                       str(batch_idx - save_batch * 3) + '.th')
            if int(epoc / save_epoc) > 3:
                if os.path.exists(args.checkpoint_dir + str(epoc - save_epoc * 3) + '.th'):
                    os.remove(args.checkpoint_dir + str(epoc - save_epoc * 3) +
                              '-' + str(batch_idx - save_batch * 3) + '.th')


def test(args, local_rank, distributed, testset):
    # basic settings
    args.batch_size = 1
    print(args)

    # get char_dict
    # with open('/data4/ydb/dataset/recognition/char_char_6489.txt')as f:
    #     chardict=f.readlines()
    # chardict = get_char_dict()
    with open('./char_char_6489.txt')as f:
        chardict = f.readlines()
    model = Model(args)
    model.to(args.device)

    # load ckpt
    with open(args.checkpoint_dir + '/last_checkpoint.txt') as f:
        lck = f.readline().strip()
    print("eval from " + lck)
    state = torch.load(args.checkpoint_dir + lck)
    model.load_state_dict(state['net'])
    model.eval()

    # data loader
    testloader = data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=0)
    fp = open('result.txt', 'w')
    t_ed = 0
    t_ac = 0

    # begin loop
    # tqdm.tqdm(testloader)
    for batch_idx, (imgs, targets) in enumerate(tqdm.tqdm(testloader, ascii=True)):
        imgs = imgs.cuda()
        targets = targets.cuda()
        seq, scores = model(imgs, targets)
        seq = seq.numpy()
        seq = seq[0, 1:]
        targets = targets.cpu().numpy()
        targets = targets[0]
        re = ''
        ret = ''
        for se in seq:
            if se > args.vocab_size:
                break
            re += chardict[int(se)].strip()
        for ta in targets:
            if int(ta) == 0:
                break
            ret += chardict[int(ta)].strip()
        ed = editdistance.eval(re, ret)
        fp.write(testset.imgs[batch_idx][0].split(
            '/')[-1] + ' ' + re + ' ' + ret +' '+ str(ed) +'\n')
        if ed == 0:
            t_ac += 1
        t_ed += ed
    print("accuracy : {}\n edit_distance: {}\n".format(
        t_ac / len(testloader), t_ed / len(testloader)))
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
    parser.add_argument("--is_training", '-t', default=False,
                        help="training or evaluation", action='store_true')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument("--ckpt", default=None,
                        help="path to save dataset", type=str,)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    args.num_layers = 2
    args.featrue_layers = 512
    args.hidden_dim = 512
    args.vocab_size = 6499
    args.out_seq_len = 30
    args.hidden_dim_de = 512
    args.embedding_size = 512
    args.batch_size = 20
    ######
    args.lr = 0.01
    args.checkpoint_dir = './model/'  # adam_lowlr/
    args.optim = ''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = 'cuda'
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # settings for multi-gpu training not implement yet
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
        pass

    # trainsform for train and test
    # TODO: data enhancement maybe in class dataset or in transform
    if args.is_training:
        transform = transforms.Compose([
            transforms.Resize((48, 320)),  # (h, w)
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((48, 320)),  # 32 280
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

    # prepare for data
    # trainset = prepare_data(args.save_dir, args.data_name, args.is_training)
    if args.is_training:
        # trainset = MyDataset('./data/icpr/crop/',
        #                  './data/icpr/char2num.txt', transform)
        # ./2423/6/96_Flowerpots_29746.jpg flowerpots
        trainset = MyDataset(
            ['icpr_train',], transform)
        sys.stdout = Logger(args.checkpoint_dir + '/log.txt')
        train(args, args.local_rank, args.distributed, trainset)
    else:
        # testset= MyDataset('/data4/ydb/dataset/recognition/imgs2_east_regions', transform=transform)
        # testset = MyDataset('./data/icpr/crop/',
        #                  './data/icpr/char2num.txt', transform)
        testset = MyDataset('SynthChinese_test', transform)
        test(args, args.local_rank, args.distributed, testset)


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
