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

# the
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


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Object Detection Training")
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
    args.is_training = False
    ######
    args.lr = 0.01
    args.checkpoint_dir = './model/'  # adam_lowlr/
    args.optim = ''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
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
    transform = transforms.Compose([
        transforms.Resize((48, 320)),  # 48 320
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    args.batch_size = 1
    print(args)

    # get char_dict
    # with open('/data4/ydb/dataset/recognition/char_char_6489.txt')as f:
    #     chardict=f.readlines()
    # chardict = get_char_dict()

    with open('./char_char_6489.txt',encoding='utf-8')as f:
        chardict = f.readlines()
    model = Model(args)
    model.to(args.device)

    # load ckpt
    with open(args.checkpoint_dir + '/last_checkpoint.txt') as f:
        lck = f.readline().strip()
    # lck = 'syn_chinese_106000.th'
    print("eval from " + lck)
    testset = MyDataset('icpr_test', transform)
    state = torch.load(args.checkpoint_dir + lck)
    model.load_state_dict(state['net'])
    print(model.recg)
    # define hook
    feat_result = []

    def get_features_hook(self, input, output):
        # # number of input:
        # print('len(input): ',len(input))
        # # number of output:
        # print('len(output): ',len(output))
        # print('###################################')
        # print(input[0].shape) # torch.Size([1, 3, 224, 224])

        # print('###################################')
        # print(output[0].shape) # torch.Size([64, 55, 55])
        feat_result.append(output.data.cpu().numpy())

    handle_feat = model.recg.attention_nn.softmax.register_forward_hook(
        get_features_hook)  # conv1
    model.eval()
    # data loader
    in_n = 176
    imgs, targets = testset[in_n]
    imgs_i = torch.unsqueeze(imgs, 0)
    targets = torch.unsqueeze(targets, 0)
    print(imgs.shape)
    imgs_i = imgs_i.cuda()
    targets = targets.cuda()
    seq, scores = model(imgs_i, targets)
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
    print("pre: " + re)
    print("gts: " + ret)

    # visual
    # imn, _ = testset.imgs[in_n]
    # im_b = np.asarray(Image.open(imn).convert('RGB'))
    tback = transforms.Compose(
        [transforms.Normalize((0, 0, 0), (1 / 0.2023, 1 / 0.1994, 1 / 0.2010)),
         transforms.Normalize((-0.4914, -0.4822, -0.4465), (1, 1, 1))]
    )
    imgt = tback(imgs)
    for i, im in enumerate(feat_result):
        # im_d = im_b.copy()
        im_d = imgt.cpu().numpy()
        im_d = im_d.transpose(1, 2, 0)
        im_d = im_d[:, :, ::-1].astype(np.float64) * 255
        im = im.reshape(6, -1)
        # print(np.max(im))
        im = im * 255
        im = im.astype(np.float64)
        # print(np.max(im))
        im = cv2.resize(im, (im_d.shape[1], im_d.shape[0]))
        im = im[..., None]
        # print(np.max(im))
        # print(np.max(im_d), np.min(im_d))
        im_d = im_d * 0.5 + im
        # print(np.max(im_d), im_d.dtype)
        # print()
        cv2.imwrite('att_vis/' + str(i) + '.jpg', im_d.astype(np.float32))
    print(ed)

    # remove hook
    handle_feat.remove()

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
