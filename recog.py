import torch.nn as nn
import math
import torch
import torch.nn.functional as F


class Recogniton(nn.Module):

    def __init__(self,args):
        super(Recogniton, self).__init__()
        self.batch_size = args.batch_size
        self.num_layers = args.num_layers
        self.featrue_layers = args.featrue_layers
        self.hidden_dim = args.hidden_dim
        self.vocab_size = args.vocab_size
        self.START_TOKEN = self.vocab_size + 1
        self.END_TOKEN = self.vocab_size + 1
        self.NULL_TOKEN = 0

        self.maxpool1 = nn.MaxPool2d((6, 1), stride=(6, 1))
        self.encode_lstm = nn.LSTM(
            self.featrue_layers, self.hidden_dim, self.num_layers)

        # nn.LSTM(10, 20, 2) featrue_l hiden_s numlayers
        # input = torch.randn(5, 3, 10) T batch feature_l
        # h_0 (2, 3, 20) numl batch hiden_s
        self.hidden_dim_de = args.hidden_dim_de
        self.out_seq_len = args.out_seq_len  # only set for eval mode, will change in train mode
        self.de_lstm_u = nn.ModuleList()
        for i in range(self.num_layers):
            self.de_lstm_u.append(nn.LSTMCell(
                self.hidden_dim, self.hidden_dim_de))
        self.embed = nn.Embedding(self.vocab_size + 2, self.hidden_dim)
        #

        self.attention_nn = Attention_nn(args)
        self.linear_out = nn.Linear(
            self.hidden_dim_de + self.featrue_layers, self.vocab_size + 2)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.loss = nn.NLLLoss(reduction='none')  # 
        print('init')

    def forward(self, conv_f, target=None):  # [-1, 512, 6, 40] | -1 30
        x = self.maxpool1(conv_f)  # [-1, 512, 1, 40]
        x = torch.squeeze(x, 2)  # [-1, 512, 40]
        x = x.permute(2, 0, 1)
        # Note that all of this 40 sequence are useful
        # TODO:get the valid ones
        self.hidden_en = (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
                          torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())
        # [40, -1, 512]
        x, self.hidden_en = self.encode_lstm(x, self.hidden_en)
        # print(self.hidden_en.shape)
        holistic_feature = x[-1]  # [-1 512]
        # decode model
        if self.training:
            self.out_seq_len = target.shape[1]
            self.init_de_lstm()
            self.gt_with_start[:, 1:] = target
            self.gt_with_start[:, 0] = self.START_TOKEN
            # mask = torch.eq(gt_with_start, 0)
            # gt_with_start[mask] = self.NULL_TOKEN
            # out -1 out_seq_len+2 numclass+1    ()
            self.forward_train(holistic_feature, conv_f, self.gt_with_start)
            out = self.out
            #
            # cal the loss
            l_target = self.l_target
            l_target[:, 1:-1] = target
            for xi in range(self.l_target.shape[0]):
                for xj in range(1, l_target.shape[1]):
                    if l_target[xi, xj] == 0:
                        l_target[xi, xj] = self.END_TOKEN
                        break
            # print(torch.argmax(out,dim=-1))
            out = out.view(-1, self.vocab_size + 2)
            l_target = l_target.view(-1)
            loss = self.loss(out, l_target)
            # print('loss',losses)
            mask = 1-torch.eq(l_target, 0)
            loss = loss[mask]
            losses=torch.mean(loss)
            # print('loss',loss)
            # print(l_target)
            # print(losses)
            return losses
        else:
            self.init_de_lstm()
            self.forward_test(holistic_feature, conv_f)
            return self.seq, self.seq_score

    def init_de_lstm(self):
        self.hidden_de = []
        for i in range(self.num_layers):
            self.hidden_de.append((torch.zeros(self.batch_size, self.hidden_dim_de).cuda(
            ), torch.zeros(self.batch_size, self.hidden_dim_de).cuda()))
        self.out = torch.zeros(self.batch_size, self.out_seq_len +
                               2, self.vocab_size + 2).cuda()
        self.gt_with_start = torch.zeros(
            self.batch_size, self.out_seq_len + 1).cuda()
        self.l_target = torch.zeros(
            self.batch_size, self.out_seq_len + 2, dtype=torch.long).cuda()
        self.seq = torch.zeros(self.batch_size, self.out_seq_len + 2)
        self.seq_score = torch.zeros(self.batch_size, self.out_seq_len + 2)

    def forward_train(self, holistic_feature, conv_f, target):
        # target batch_size*L   L<max_w
        for t in range(self.out_seq_len + 2):
            if t == 0:
                xt = holistic_feature
            else:
                it = target[:, t - 1]
                it = it.view(-1).to(torch.long).to('cuda')
                xt = self.embed(it)

            for i in range(self.num_layers):
                if i == 0:
                    inp = xt
                else:
                    inp = self.hidden_de[i - 1][0]
                self.hidden_de[i] = self.de_lstm_u[
                    i](inp, self.hidden_de[i])  # -1 512
            h = self.hidden_de[-1][0]
            att, attw = self.attention_nn(conv_f, h)  # -1 512 |  -1  h*w
            tmpcat = torch.cat((h, att), -1)
            scores = self.logsoftmax(self.linear_out(tmpcat))  # -1 vc_s+1
            self.out[:, t, :] = scores

    def forward_test(self, holistic_feature, conv_f):
        for t in range(self.out_seq_len + 2):
            if t == 0:
                xt = holistic_feature
            elif t == 1:
                it = torch.zeros(self.batch_size).fill_(
                    self.START_TOKEN).to(torch.long).to('cuda')
                xt = self.embed(it)
            else:
                it = self.seq[:, t - 1]
                it = it.view(-1).to(torch.long).to('cuda')
                xt = self.embed(it)
            for i in range(self.num_layers):
                if i == 0:
                    inp = xt
                else:
                    inp = self.hidden_de[i - 1][0]
                self.hidden_de[i] = self.de_lstm_u[
                    i](inp, self.hidden_de[i])  # -1 512
            h = self.hidden_de[-1][0]
            att, attw = self.attention_nn(conv_f, h)  # -1 512 |  -1  h*w
            tmpcat = torch.cat((h, att), -1)
            scores = self.logsoftmax(self.linear_out(tmpcat))  # -1 vc_s+1
            idxscore, idx = torch.max(scores, 1)
            self.seq[:, t] = idx
            self.seq_score[:, t] = idxscore


class Attention_nn(nn.Module):
    # TODO:get the valid ones

    def __init__(self,args):
        super(Attention_nn, self).__init__()
        self.batch_size = args.batch_size
        self.featrue_layers = args.featrue_layers
        self.hidden_dim_de = args.hidden_dim_de
        self.embedding_size = args.embedding_size
        self.max_h = args.max_h
        self.max_w = args.max_w
        self.conv_h = nn.Linear(self.hidden_dim_de, self.embedding_size)
        self.conv_f = nn.Conv2d(self.featrue_layers,
                                self.embedding_size, kernel_size=3, padding=1)
        self.conv_att = nn.Linear(self.embedding_size, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, conv_f, h):
        g_em = self.conv_h(h)
        g_em = g_em.view(self.batch_size, -1, 1)
        g_em = g_em.repeat(1, 1, self.max_h * self.max_w)  # -1 512 h*w
        g_em = g_em.permute(0, 2, 1)
        x_em = self.conv_f(conv_f)
        x_em = x_em.view(self.batch_size, -1, self.max_h * self.max_w)
        x_em = x_em.permute(0, 2, 1)
        feat = self.dropout(torch.tanh(x_em + g_em))  # -1 h*w 512
        e = self.conv_att(feat.view(-1, self.embedding_size))  # -1*h*w 1
        alpha = self.softmax(e.view(-1, self.max_h * self.max_w))  # -1  h*w
        alpha2 = alpha.view(-1, 1, self.max_h * self.max_w)  # -1 1 h*w
        orgfeat_embed = conv_f.view(-1, self.featrue_layers,
                                    self.max_h * self.max_w)
        orgfeat_embed = orgfeat_embed.permute(0, 2, 1)  # -1 h*w 512
        att_out = torch.matmul(alpha2, orgfeat_embed)
        att_out = att_out.view(-1, self.featrue_layers)  # -1 512
        return att_out, alpha
