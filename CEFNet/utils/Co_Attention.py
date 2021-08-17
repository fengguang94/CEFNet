import torch
from torch import nn
from torch.nn import functional as F



class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class ACM(nn.Module):
    def __init__(self, low_in_channels, high_in_channels, key_channels, value_channels, out_channels=None, scale=1, norm_type=None,psp_size=(1,3,6,8)):
        super(ACM, self).__init__()
        self.scale = scale
        self.in_channels = low_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = high_in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key_img = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_key_lang = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_query_img = nn.Sequential(
            nn.Conv2d(in_channels=high_in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_query_lang = nn.Sequential(
            nn.Conv2d(in_channels=high_in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_value_img = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.f_value_lang = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W_img = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        self.W_lang = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                               kernel_size=1, stride=1, padding=0)

        self.psp = PSPModule(psp_size)
        # nn.init.constant_(self.W.weight, 0)
        # nn.init.constant_(self.W.bias, 0)

    def forward(self, img_feats, lang_feats):
        batch_size, h, w = img_feats.size(0), img_feats.size(2), img_feats.size(3)

        value_img = self.psp(self.f_value_img(img_feats)).permute(0, 2, 1)
        value_lang = self.psp(self.f_value_lang(lang_feats)).permute(0, 2, 1)

        query_img = self.f_query_img(img_feats).view(batch_size, self.key_channels, -1).permute(0, 2, 1)
        query_lang = self.f_query_lang(lang_feats).view(batch_size, self.key_channels, -1).permute(0, 2, 1)

        key_img = self.psp(self.f_key_img(img_feats))
        key_lang = self.psp(self.f_key_lang(lang_feats))

        sim_map_img = torch.matmul(query_img, key_img)
        sim_map_img = (self.key_channels ** -.5) * sim_map_img

        sim_map_lang = torch.matmul(query_lang, key_lang)
        sim_map_lang = (self.key_channels ** -.5) * sim_map_lang

        sim_map = F.softmax(sim_map_img+sim_map_lang, dim=-1)

        context_img = torch.matmul(sim_map, value_img)
        context_img = context_img.permute(0, 2, 1).contiguous()
        context_img = context_img.view(batch_size, self.value_channels, *img_feats.size()[2:])
        context_img = self.W_img(context_img)

        context_lang = torch.matmul(sim_map, value_lang)
        context_lang = context_lang.permute(0, 2, 1).contiguous()
        context_lang = context_lang.view(batch_size, self.value_channels, *img_feats.size()[2:])
        context_lang = self.W_lang(context_lang)

        return context_img, context_lang

class VCM(nn.Module):
    def __init__(self, in_channel=512, out_channel=512, all_dim=40 * 40):
        super(VCM, self).__init__()
        self.linear_e = nn.Conv2d(in_channel, 256, kernel_size=1, padding=0, bias=False)
        self.linear_q = nn.Conv2d(in_channel, 256, kernel_size=1, padding=0, bias=False)
        #self.channel = all_channel
        self.dim = all_dim
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # init.xavier_normal(m.weight.data)
                # m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, exemplar, query):

        fea_size = query.size()[2:]
        all_dim = fea_size[0] * fea_size[1]
        query_corr = self.linear_q(query)
        exemplar_corr = self.linear_e(exemplar)  #
        exemplar_flat = exemplar_corr.view(-1, exemplar_corr.size()[1], all_dim)  # N,C,H*W
        query_flat = query_corr.view(-1, query_corr.size()[1], all_dim)
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num
        A = torch.bmm(exemplar_t, query_flat)
        A1 = F.softmax(A.clone(), dim=1).transpose(1,2) #
        B = F.softmax(torch.transpose(A, 1, 2), dim=1).transpose(1,2)
        exemplar_flat = exemplar.view(-1, exemplar.size()[1], all_dim)  # N,C,H*W
        query_flat = query.view(-1, query.size()[1], all_dim)
        query_att = torch.bmm(exemplar_flat, A1).contiguous()
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        input1_att = exemplar_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        input2_att = query_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        input1_att = self.conv1(input1_att)
        input2_att = self.conv2(input2_att)
        return input1_att, input2_att  # shape:
