from collections import OrderedDict
import torch
from torch import nn
from torchsummary import summary
import resnet
import recog
import os
import argparse


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv_layer = resnet.ResNet(31)
        self.recg = recog.Recogniton()

    def forward(self, image, target=None):
        conv_f = self.conv_layer(image)
        if self.training:
            out = self.recg(conv_f, target)
        else:
            seq, seq_score = self.recg(conv_f)  # 2 32 | 2 32
            return seq, seq_score
        return conv_f


def train(cfg, local_rank, distributed):
    model = Model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    params = []
    for key, value in model.named_parameters():
        # print(key)
        if not value.requires_grad:
            continue
        lr = 0.02  # cfg.SOLVER.BASE_LR
        weight_decay = 0.0001  # cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = 0.02 * 2  # cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = 0  # cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # cfg.SOLVER.MOMENTUM
    optimizer = torch.optim.SGD(params, lr, momentum=0.9)
    model.train()
    # data loader
    img = torch.rand(2, 3, 48, 160).cuda()
    target = torch.zeros(2, 30).cuda()
    target[:, :10] = 1
    target[:, 10:19] = 2
    model.eval()
    for i in range(1000):
    #     losses = model(img, target)
    #     # cal loss
    #     optimizer.zero_grad()
    #     losses.backward(retain_graph=True)
    #     optimizer.step()
    #     if i % 1 == 0:
    #         print(i, losses.item())
    
        seq, seq_score = model(img, target)
    print(seq, seq_score)


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

    num_gpus = int(os.environ["WORLD_SIZE"]
                   ) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    cfg = 0
    model = train(cfg, args.local_rank, args.distributed)


if __name__ == '__main__':
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
    main()
