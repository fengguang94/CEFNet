import torch, torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List


class ConvBNRelu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, G=32, use_relu=True):
        super(ConvBNRelu, self).__init__()
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.use_relu:
            self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x

def generate_coord(batch, height, width):
    # coord = Variable(torch.zeros(batch,8,height,width).cuda())
    #print(batch, height, width)
    xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    #print(batch, height, width)
    xv_min = (xv.float()*2 - width)/width
    yv_min = (yv.float()*2 - height)/height
    xv_max = ((xv+1).float()*2 - width)/width
    yv_max = ((yv+1).float()*2 - height)/height
    xv_ctr = (xv_min+xv_max)/2
    yv_ctr = (yv_min+yv_max)/2
    hmap = torch.ones(height,width)*(1./height)
    wmap = torch.ones(height,width)*(1./width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch,1,1,1)
    return coord

class WordVisualAttention(nn.Module):
  def __init__(self, input_dim):
    super(WordVisualAttention, self).__init__()
    # initialize pivot
    self.visual = nn.Conv2d(input_dim, input_dim, kernel_size=1)

  def forward(self, context, visual, input_labels):
    """
    Inputs:
    - context : Variable float (batch, seq_len, input_dim)
    - embedded: Variable float (batch, seq_len, word_vec_size)
    - input_labels: Variable long (batch, seq_len)
    Outputs:
    - attn    : Variable float (batch, seq_len)
    - weighted_emb: Variable float (batch, word_vec_size)
    """
    visual = self.visual(visual)
    b_size, n_channel, h, w = visual.shape
    visual = visual.view(b_size, n_channel, h*w)
    attn = torch.bmm(context, visual)
    attn = F.softmax(attn, dim=1)  # (batch, seq_len), attn.sum(1) = 1.

    # mask zeros
    is_not_zero = (input_labels!=0).float()
    is_not_zero = is_not_zero.view(is_not_zero.size(0), is_not_zero.size(1), 1).repeat(1, 1, h*w)
    attn = attn * is_not_zero
    attn = attn / attn.sum(1).view(attn.size(0), 1, attn.size(2)).repeat(1, attn.size(1), 1)

    # compute weighted lang
    weighted_emb = torch.bmm(context.permute(0, 2, 1), attn)
    weighted_emb = weighted_emb.view(weighted_emb.size(0), weighted_emb.size(1), h, w)

    return weighted_emb

class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, word_vec_size, hidden_size, bidirectional=False,
               input_dropout_p=0, dropout_p=0, n_layers=1, rnn_type='lstm', variable_lengths=True):
        super(RNNEncoder, self).__init__()
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, word_embedding_size)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.mlp = nn.Sequential(nn.Linear(word_embedding_size, word_vec_size),
                                 nn.ReLU())
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(word_vec_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)
        self.num_dirs = 2 if bidirectional else 1

    def forward(self, input_labels):
        """
        Inputs:
        - input_labels: Variable long (batch, seq_len)
        Outputs:
        - output  : Variable float (batch, max_len, hidden_size * num_dirs)
        - hidden  : Variable float (batch, num_layers * num_dirs * hidden_size)
        - embedded: Variable float (batch, max_len, word_vec_size)
        """
        if self.variable_lengths:
            input_lengths = (input_labels!=0).sum(1)  # Variable (batch, )

            # make ixs
            input_lengths_list = input_lengths.data.cpu().numpy().tolist()
            sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist() # list of sorted input_lengths
            sort_ixs = np.argsort(input_lengths_list)[::-1].tolist() # list of int sort_ixs, descending
            s2r = {s: r for r, s in enumerate(sort_ixs)} # O(n)
            recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]  # list of int recover ixs
            assert max(input_lengths_list) == input_labels.size(1)

            # move to long tensor
            sort_ixs = input_labels.data.new(sort_ixs).long()  # Variable long
            recover_ixs = input_labels.data.new(recover_ixs).long()  # Variable long

            # sort input_labels by descending order
            input_labels = input_labels[sort_ixs]

        # embed
        embedded = self.embedding(input_labels)  # (n, seq_len, word_embedding_size)
        embedded = self.input_dropout(embedded)  # (n, seq_len, word_embedding_size)
        embedded = self.mlp(embedded)            # (n, seq_len, word_vec_size)
        if self.variable_lengths:
            sorted_input_lengths_list = torch.as_tensor(sorted_input_lengths_list, dtype=torch.int64).cpu()
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, torch.as_tensor(sorted_input_lengths_list, dtype=torch.int64).cpu(), batch_first=True)

        # forward rnn
        self.rnn.flatten_parameters() #不加这一行会报warning
        output, hidden = self.rnn(embedded)

        # recover
        if self.variable_lengths:

            # embedded (batch, seq_len, word_vec_size)
            embedded, _ = nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)
            embedded = embedded[recover_ixs]

            # recover rnn
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # (batch, max_len, hidden)
            output = output[recover_ixs]

            # # recover hidden
            # if self.rnn_type == 'lstm':
            #     hidden = hidden[0]
            # hidden = hidden[:, recover_ixs, :]  # (num_layers * num_dirs, batch, hidden_size)
            # hidden = hidden.transpose(0, 1).contiguous()  # (batch, num_layers * num_dirs, hidden_size)
            # hidden = hidden.view(hidden.size(0), -1)  # (batch, num_layers * num_dirs * hidden_size)
        sent_output = []
        for ii in range(output.shape[0]):
            sent_output.append(output[ii, int(input_lengths_list[ii] - 1), :])

        return output, torch.stack(sent_output, dim=0), embedded

class Bottleneck(nn.Module):
    """ Adapted from torchvision.models.resnet """
    expansion = 4

    def __init__(self, inplanes, planes, middle, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvBNRelu(inplanes, middle, kernel_size=1, stride=1, padding=0, use_relu=True)
        self.conv2 = ConvBNRelu(middle, middle, kernel_size=3, stride=1, padding=1, use_relu=True)
        self.conv3 = ConvBNRelu(middle, planes, kernel_size=1, stride=1, padding=0, use_relu=False)
        self.downsample = ConvBNRelu(inplanes, planes, kernel_size=1, stride=1, padding=0, use_relu=False)
        self.relu = nn.PReLU()
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
