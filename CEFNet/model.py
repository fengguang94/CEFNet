import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from data.config import cfg
from utils import timer
from util_fun import RNNEncoder, WordVisualAttention, generate_coord
from utils.Co_Attention import ACM, VCM
import time

# These are in BGR and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD   = (57.38, 57.12, 58.40)
means = np.array(MEANS)[np.newaxis,:,np.newaxis,np.newaxis]
std = np.array(STD)[np.newaxis,:,np.newaxis,np.newaxis]

def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=False)
class SqueezeBodyEdge(nn.Module):
    def __init__(self, inplane, norm_layer):
        super(SqueezeBodyEdge, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2, padding=1),
            norm_layer(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2, padding=1),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )

        self.flow_make = nn.Conv2d(inplane *2 , inplane, kernel_size=3, padding=1, bias=False)
        self.pooling = torch.nn.AdaptiveAvgPool2d(output_size=(1))
        # 3 * 2 affine矩阵的回归量
        self.fc_loc = nn.Sequential(
            nn.Linear(inplane, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # 使用身份转换初始化权重/偏差
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        size = x.size()[2:]
        seg_down = self.down(x)
        seg_down = F.upsample(seg_down, size=size, mode="bilinear", align_corners=False)
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))
        flow = self.pooling(flow)
        flow = flow.view(x.size()[0], -1).contiguous()
        theta = self.fc_loc(flow)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        seg_flow_warp = F.grid_sample(x, grid)
        seg_edge = x - seg_flow_warp
        return seg_flow_warp, seg_edge

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + torch.cat([flow.permute(0, 2, 3, 1)[:,:,:,0:1]/out_w, flow.permute(0, 2, 3, 1)[:,:,:,1:2]/out_h], dim=3)
        #grid = grid + flow.permute(0, 2, 3, 1) # / norm

        output = F.grid_sample(input, grid)
        return output

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

class model(nn.Module):
    def __init__(self):
        super().__init__()

        feats = list(models.wide_resnet101_2(pretrained=True).children())
        self.img_bn = nn.BatchNorm2d(3)
        self.scale1 = nn.Sequential(*feats[0:3])
        self.scale2 = nn.Sequential(*feats[3:5])
        self.scale3 = feats[5]
        self.scale4 = feats[6]
        self.scale5 = feats[7]
        ####################################
        self.wordvisual3 = WordVisualAttention(512)
        self.wordvisual4 = WordVisualAttention(512)
        self.wordvisual5 = WordVisualAttention(512)
        ####################################
        # self.attention3 = ACM(low_in_channels=512, high_in_channels=512, key_channels=256,
        #                                    value_channels=256, out_channels=512)
        # self.attention4 = ACM(low_in_channels=512, high_in_channels=512, key_channels=256,
        #                                     value_channels=256, out_channels=512)
        # self.attention5 = ACM(low_in_channels=512, high_in_channels=512, key_channels=256,
        #                                     value_channels=256, out_channels=512)
        self.attention3 = VCM(in_channel=512, out_channel=512, all_dim=40 * 40)
        self.attention4 = VCM(in_channel=512, out_channel=512, all_dim=20 * 20)
        self.attention5 = VCM(in_channel=512, out_channel=512, all_dim=20 * 20)
        for m in self.scale4:
            m.conv2.stride = (2, 2)
            for n in m.downsample:
                if isinstance(n, nn.Conv2d):
                    n.stride = (2, 2)
            break
        for m in self.scale4:
            m.conv2.dilation = (1, 1)
            m.conv2.padding = (1, 1)
        for m in self.scale5:
            m.conv2.stride = (1, 1)
            for n in m.downsample:
                if isinstance(n, nn.Conv2d):
                    n.stride = (1, 1)
            break
        for m in self.scale5:
            m.conv2.dilation = (1, 1)
            m.conv2.padding = (1, 1)
        ################################################################################################################
        cfg.dcm_c = 512
        self.lang_c3_1 = ConvBNRelu(512, 512, kernel_size=3, stride=1, padding=1)
        self.lang_c3_2 = ConvBNRelu(512 * 2 + 8, cfg.dcm_c, kernel_size=3, stride=1, padding=1)
        self.lang_c3_3 = ConvBNRelu(cfg.dcm_c*2, 512, kernel_size=3, stride=1, padding=1)
        ##################
        self.lang_c4_1 = ConvBNRelu(1024, 512, kernel_size=3, stride=1, padding=1)
        self.lang_c4_2 = ConvBNRelu(512 * 2 + 8, cfg.dcm_c, kernel_size=3, stride=1, padding=1)
        self.lang_c4_3 = ConvBNRelu(cfg.dcm_c*2, 1024, kernel_size=3, stride=1, padding=1)
        ##################
        self.lang_c5_1 = ConvBNRelu(2048, 512, kernel_size=3, stride=1, padding=1)
        self.lang_c5_2 = ConvBNRelu(512 * 2 + 8, cfg.dcm_c, kernel_size=3, stride=1, padding=1)
        self.lang_c5_3 = ConvBNRelu(cfg.dcm_c*2, 2048, kernel_size=3, stride=1, padding=1)
        ##################
        self.reduced_c5 = ConvBNRelu(2048, 512, kernel_size=3, stride=1, padding=1)
        self.reduced_c4 = ConvBNRelu(1024, 256, kernel_size=3, stride=1, padding=1)
        self.reduced_c3 = ConvBNRelu(512, 128, kernel_size=3, stride=1, padding=1)
        self.reduced_c2 = ConvBNRelu(256, 64, kernel_size=3, stride=1, padding=1)
        self.reduced_c1 = ConvBNRelu(64, 32, kernel_size=3, stride=1, padding=1)
        ################################################################################################################
        ##################
        self.squeeze_body_edge5 = SqueezeBodyEdge(256, nn.BatchNorm2d)
        self.squeeze_body_edge4 = SqueezeBodyEdge(128, nn.BatchNorm2d)
        self.squeeze_body_edge3 = SqueezeBodyEdge(64, nn.BatchNorm2d)
        self.squeeze_body_edge2 = SqueezeBodyEdge(32, nn.BatchNorm2d)
        # fusion different edge part
        self.edge_fusion5 = nn.Conv2d(256 * 2, 256, 1, bias=False)
        self.edge_fusion4 = nn.Conv2d(128 * 2, 128, 1, bias=False)
        self.edge_fusion3 = nn.Conv2d(64 * 2, 64, 1, bias=False)
        self.edge_fusion2 = nn.Conv2d(32 * 2, 32, 1, bias=False)
        ##################
        self.sigmoid_edge = nn.Sigmoid()
        self.edge_out1 = nn.Sequential(
            ConvBNRelu(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=1, bias=False))
        self.dsn_seg_body1 = nn.Sequential(
            ConvBNRelu(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=1, bias=False))
        ##################
        self.edge_out2 = nn.Sequential(
            ConvBNRelu(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=1, bias=False))
        self.dsn_seg_body2 = nn.Sequential(
            ConvBNRelu(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=1, bias=False))
        self.final_seg2 = nn.Sequential(
            ConvBNRelu(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=1, bias=False))
        ##################
        self.edge_out3 = nn.Sequential(
            ConvBNRelu(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, bias=False))
        self.dsn_seg_body3 = nn.Sequential(
            ConvBNRelu(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, bias=False))
        self.final_seg3 = nn.Sequential(
            ConvBNRelu(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=1, bias=False))
        ##################
        self.edge_out4 = nn.Sequential(
            ConvBNRelu(256, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 1, kernel_size=1, bias=False))
        self.dsn_seg_body4 = nn.Sequential(
            ConvBNRelu(256, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 1, kernel_size=1, bias=False))
        self.final_seg4 = nn.Sequential(
            ConvBNRelu(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, bias=False))
        ##################
        self.output5 = ConvBNRelu(512, 256, kernel_size=3, stride=1, padding=1)
        self.output4 = ConvBNRelu(512, 128, kernel_size=3, stride=1, padding=1)
        self.output3 = ConvBNRelu(256, 64, kernel_size=3, stride=1, padding=1)
        self.output2 = ConvBNRelu(128, 32, kernel_size=3, stride=1, padding=1)
        self.output1 = nn.Sequential(ConvBNRelu(64, 32, kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(32, 1, kernel_size=3, padding=1))
        ################################################################################################################
        self.textdim, self.embdim, self.emb_size = 1024, 512, 512
        jemb_drop_out = 0.0
        self.textmodel = RNNEncoder(vocab_size=12112,  #  'referit' 8803;  else 12112
                                    word_embedding_size=self.embdim,
                                    word_vec_size=self.textdim // 2,
                                    hidden_size=self.textdim // 2,
                                    bidirectional=True,
                                    input_dropout_p=0.0,
                                    dropout_p=0.0,
                                    rnn_type='gru', # lstm gru
                                    variable_lengths=True)
        self.mapping_lang = torch.nn.Sequential(
            nn.Linear(self.embdim, self.emb_size),
            nn.BatchNorm1d(self.emb_size),
            nn.ReLU(),
            nn.Dropout(jemb_drop_out)
        )
        self.c3_lang = torch.nn.Sequential(
            nn.Linear(self.emb_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU())
        self.c4_lang = torch.nn.Sequential(
            nn.Linear(self.emb_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU())
        self.c5_lang = torch.nn.Sequential(
            nn.Linear(self.emb_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU())
        self.mapping_hT = torch.nn.Sequential(
            nn.Linear(self.textdim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        if cfg.freeze_bn:
            self.freeze_bn()
        ################################################################################################################

    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)

        # For backward compatability, remove these (the new variable is called layers)
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]

            # Also for backward compatibility with v1.0 weights, do this check
            if key.startswith('fpn.downsample_layers.'):
                if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
                    del state_dict[key]
        self.load_state_dict(state_dict)

    # def init_weights(self, backbone_path):
    #     """ Initialize weights for training. """
    #     # Initialize the backbone with the pretrained weights.
    #     self.backbone.init_backbone(backbone_path)
    #
    #     conv_constants = getattr(nn.Conv2d(1, 1, 1), '__constants__')
    #
    #     # Quick lambda to test if one list contains the other
    #     def all_in(x, y):
    #         for _x in x:
    #             if _x not in y:
    #                 return False
    #         return True
    #
    #     # Initialize the rest of the conv layers with xavier
    #     for name, module in self.named_modules():
    #         # See issue #127 for why we need such a complicated condition if the module is a WeakScriptModuleProxy
    #         # Broke in 1.3 (see issue #175), WeakScriptModuleProxy was turned into just ScriptModule.
    #         # Note that this might break with future pytorch updates, so let me know if it does
    #         is_script_conv = 'Script' in type(module).__name__ \
    #                          and all_in(module.__dict__['_constants_set'], conv_constants) \
    #                          and all_in(conv_constants, module.__dict__['_constants_set'])
    #
    #         is_conv_layer = isinstance(module, nn.Conv2d) or is_script_conv
    #         if isinstance(module, nn.Linear):
    #             print(name)
    #
    #         if is_conv_layer and module not in self.backbone.backbone_modules:
    #             nn.init.xavier_uniform_(module.weight.data)
    #
    #             if module.bias is not None:
    #                 if cfg.use_focal_loss and 'cls_logits' in name:
    #                     if not cfg.use_sigmoid_focal_loss:
    #                         # Initialize the last layer as in the focal loss paper.
    #                         # Because we use softmax and not sigmoid, I had to derive an alternate expression
    #                         # on a notecard. Define pi to be the probability of outputting a foreground detection.
    #                         # Then let z = sum(exp(x)) - exp(x_0). Finally let c be the number of foreground classes.
    #                         # Chugging through the math, this gives us
    #                         #   x_0 = log(z * (1 - pi) / pi)    where 0 is the background class
    #                         #   x_i = log(z / c)                for all i > 0
    #                         # For simplicity (and because we have a degree of freedom here), set z = 1. Then we have
    #                         #   x_0 =  log((1 - pi) / pi)       note: don't split up the log for numerical stability
    #                         #   x_i = -log(c)                   for all i > 0
    #                         module.bias.data[0] = np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
    #                         module.bias.data[1:] = -np.log(module.bias.size(0) - 1)
    #                     else:
    #                         module.bias.data[0] = -np.log(cfg.focal_loss_init_pi / (1 - cfg.focal_loss_init_pi))
    #                         module.bias.data[1:] = -np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
    #                 else:
    #                     module.bias.data.zero_()

    def train(self, mode=True):
        super().train(mode)

        if cfg.freeze_bn:
            self.freeze_bn()

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

    def forward(self, x, word_id):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        _, _, img_h, img_w = x.size()
        cfg._tmp_img_h = img_h
        cfg._tmp_img_w = img_w
        ################################################################################################################
        with timer.env('lstm'):
            max_len = (word_id != 0).sum(1).max().item()
            word_id = word_id[:, :max_len]
            _, hidden, context = self.textmodel(word_id)
            b_size, l_length, emd_dim = context.shape
            context = context.view(-1, emd_dim)
            flang = self.mapping_lang(context)
            lang_c3 = self.c3_lang(flang)
            lang_c4 = self.c4_lang(flang)
            lang_c5 = self.c5_lang(flang)
            ######
            lang_c3 = F.normalize(lang_c3.view(b_size, l_length, -1), p=2, dim=2)
            lang_c4 = F.normalize(lang_c4.view(b_size, l_length, -1), p=2, dim=2)
            lang_c5 = F.normalize(lang_c5.view(b_size, l_length, -1), p=2, dim=2)
            HT = self.mapping_hT(hidden)
            HT = F.normalize(HT, p=2, dim=1)
            ######
        ################################################################################################################
        with timer.env('backbone'):
            x = self.img_bn(x)
            #x = (x - torch.FloatTensor(means).cuda())/torch.FloatTensor(std).cuda()
            c1 = self.scale1(x)
            c2 = self.scale2(c1)
            c3 = self.scale3(c2)
            ####################################
            coord = generate_coord(c3.size(0), c3.size(2), c3.size(3))
            HT_c3_tile = HT.view(HT.size(0), HT.size(1), 1, 1).repeat(1, 1, c3.size(2), c3.size(3))
            temp_c3 = F.normalize(self.lang_c3_1(c3), p=2, dim=1)
            temp_c3 = F.normalize(self.lang_c3_2(torch.cat([temp_c3, HT_c3_tile, coord], dim=1)))
            c3_tile = self.wordvisual3(lang_c3, temp_c3, word_id)
            temp_c3, c3_tile = self.attention3(temp_c3, c3_tile)
            temp_c3 = F.normalize(torch.cat([c3_tile, temp_c3], dim=1))
            c3 = F.normalize(c3, p=2, dim=1) + F.normalize(self.lang_c3_3(temp_c3), p=2, dim=1)
            ####################################
            c4 = self.scale4(c3)
            ####################################
            coord = generate_coord(c4.size(0), c4.size(2), c4.size(3))
            HT_c4_tile = HT.view(HT.size(0), HT.size(1), 1, 1).repeat(1, 1, c4.size(2), c4.size(3))
            temp_c4 = F.normalize(self.lang_c4_1(c4), p=2, dim=1)
            temp_c4 = F.normalize(self.lang_c4_2(torch.cat([temp_c4, HT_c4_tile, coord], dim=1)))
            c4_tile = self.wordvisual4(lang_c4, temp_c4, word_id)
            temp_c4, c4_tile = self.attention4(temp_c4, c4_tile)
            temp_c4 = F.normalize(torch.cat([c4_tile, temp_c4], dim=1))
            c4 = F.normalize(c4, p=2, dim=1) + F.normalize(self.lang_c4_3(temp_c4), p=2, dim=1)
            ####################################
            c5 = self.scale5(c4)
            ####################################
            coord = generate_coord(c5.size(0), c5.size(2), c5.size(3))
            HT_c5_tile = HT.view(HT.size(0), HT.size(1), 1, 1).repeat(1, 1, c5.size(2), c5.size(3))
            temp_c5 = F.normalize(self.lang_c5_1(c5), p=2, dim=1)
            temp_c5 = F.normalize(self.lang_c5_2(torch.cat([temp_c5, HT_c5_tile, coord], dim=1)))
            c5_tile = self.wordvisual5(lang_c5, temp_c5, word_id)
            temp_c5, c5_tile = self.attention5(temp_c5, c5_tile)
            temp_c5 = F.normalize(torch.cat([c5_tile, temp_c5], dim=1))
            c5 = F.normalize(c5, p=2, dim=1) + F.normalize(self.lang_c5_3(temp_c5), p=2, dim=1)
            ####################################
        ################################################################################################################
        with timer.env('FPN'):
            dem1 = self.reduced_c1(c1)
            dem2 = self.reduced_c2(c2)
            dem3 = self.reduced_c3(c3)
            dem4 = self.reduced_c4(c4)
            dem5 = self.reduced_c5(c5)
            output5 = self.output5(dem5)
            ####################################
            seg_body5, seg_edge5 = self.squeeze_body_edge5(output5)
            seg_edge5 = self.edge_fusion5(torch.cat([Upsample(seg_edge5, dem4.shape[2:]), dem4], dim=1))
            seg_out5 = seg_edge5 + Upsample(seg_body5, seg_edge5.shape[2:])
            seg_out4 = self.output4(torch.cat([Upsample(output5, seg_out5.shape[2:]), seg_out5], dim=1))
            if self.training:
                seg_edge_out4 = self.sigmoid_edge(Upsample(self.edge_out4(seg_edge5), x.shape[2:]))
                seg_final_out4 = Upsample(self.final_seg4(seg_out4), x.shape[2:])
                seg_body_out4 = Upsample(self.dsn_seg_body4(seg_body5), x.shape[2:])
            tmp_seg_out4 = seg_out4.clone()
            ####################################
            seg_body4, seg_edge4 = self.squeeze_body_edge4(seg_out4)
            seg_edge4 = self.edge_fusion4(torch.cat([Upsample(seg_edge4, dem3.shape[2:]), dem3], dim=1))
            seg_out4 = seg_edge4 + Upsample(seg_body4, seg_edge4.shape[2:])
            seg_out3 = self.output3(torch.cat([Upsample(tmp_seg_out4, seg_out4.shape[2:]), seg_out4], dim=1))
            if self.training:
                seg_edge_out3 = self.sigmoid_edge(Upsample(self.edge_out3(seg_edge4), x.shape[2:]))
                seg_final_out3 = Upsample(self.final_seg3(seg_out3), x.shape[2:])
                seg_body_out3 = Upsample(self.dsn_seg_body3(seg_body4), x.shape[2:])
            tmp_seg_out3 = seg_out3.clone()
            ####################################
            seg_body3, seg_edge3 = self.squeeze_body_edge3(seg_out3)
            seg_edge3 = self.edge_fusion3(torch.cat([Upsample(seg_edge3, dem2.shape[2:]), dem2], dim=1))
            seg_out3 = seg_edge3 + Upsample(seg_body3, seg_edge3.shape[2:])
            seg_out2 = self.output2(torch.cat([Upsample(tmp_seg_out3, seg_out3.shape[2:]), seg_out3], dim=1))
            if self.training:
                seg_edge_out2 = self.sigmoid_edge(Upsample(self.edge_out2(seg_edge3), x.shape[2:]))
                seg_final_out2 = Upsample(self.final_seg2(seg_out2), x.shape[2:])
                seg_body_out2 = Upsample(self.dsn_seg_body2(seg_body3), x.shape[2:])
            tmp_seg_out2 = seg_out2.clone()
            ####################################
            seg_body2, seg_edge2 = self.squeeze_body_edge2(seg_out2)
            seg_edge2 = self.edge_fusion2(torch.cat([Upsample(seg_edge2, dem1.shape[2:]), dem1], dim=1))
            seg_out2 = seg_edge2 + Upsample(seg_body2, seg_edge2.shape[2:])
            seg_out1 = self.output1(torch.cat([Upsample(tmp_seg_out2, seg_out2.shape[2:]), seg_out2], dim=1))
            seg_final_out1 = Upsample(seg_out1, x.shape[2:])
            hot_map = self.output1[0].conv(torch.cat([Upsample(tmp_seg_out2, seg_out2.shape[2:]), seg_out2], dim=1))
            if self.training:
                seg_edge_out1 = self.sigmoid_edge(Upsample(self.edge_out1(seg_edge2), x.shape[2:]))
                seg_body_out1 = Upsample(self.dsn_seg_body1(seg_body2), x.shape[2:])
            ####################################
            ####################################
            ################################################################################################################
            # return [torch.sigmoid(output), torch.sigmoid(deep3), torch.sigmoid(deep4), torch.sigmoid(deep5)]
        if self.training:
            return [seg_final_out1, seg_body_out1, seg_edge_out1, seg_final_out2, seg_body_out2, seg_edge_out2,
                    seg_final_out3, seg_body_out3, seg_edge_out3, seg_final_out4, seg_body_out4, seg_edge_out4]
        else:
            return seg_final_out1#, torch.mean(hot_map, dim=1)