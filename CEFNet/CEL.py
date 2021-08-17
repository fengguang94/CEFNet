from torch import nn
import torch
import torch.nn.functional as F

class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        print("You are using `CEL`!")
        self.eps = 1e-6
        #self.loss = nn.BCEWithLogitsLoss(reduction="none")
        #self.loss = nn.CrossEntropyLoss(reduction=mean)


    def forward(self, pred, masks):
        target = masks[:, 0:1, :, :]
        edge_m = masks[:, 1:2, :, :]
        b, c, h, w = target.size()
        seg_final_out1, seg_body_out1, seg_edge_out1, seg_final_out2, seg_body_out2, seg_edge_out2, \
        seg_final_out3, seg_body_out3, seg_edge_out3, seg_final_out4, seg_body_out4, seg_edge_out4 = pred
        ############################################################################################################
        losses = {'M1': F.binary_cross_entropy_with_logits(input=seg_final_out1, target=target, reduction='mean')}
        losses['B1'] = F.binary_cross_entropy_with_logits(input=seg_body_out1, target=target, reduction='mean')
        losses['E1'] = F.binary_cross_entropy(input=seg_edge_out1, target=edge_m, reduction='mean')
        ######
        losses['M2'] = F.binary_cross_entropy_with_logits(input=seg_final_out2, target=target, reduction='mean')
        losses['B2'] = F.binary_cross_entropy_with_logits(input=seg_body_out2, target=target, reduction='mean')
        losses['E2'] = F.binary_cross_entropy(input=seg_edge_out2, target=edge_m, reduction='mean')
        ######
        losses['M3'] = F.binary_cross_entropy_with_logits(input=seg_final_out3, target=target, reduction='mean')
        losses['B3'] = F.binary_cross_entropy_with_logits(input=seg_body_out3, target=target, reduction='mean')
        losses['E3'] = F.binary_cross_entropy(input=seg_edge_out3, target=edge_m, reduction='mean')
        ######
        losses['M4'] = F.binary_cross_entropy_with_logits(input=seg_final_out4, target=target, reduction='mean')
        losses['B4'] = F.binary_cross_entropy_with_logits(input=seg_body_out4, target=target, reduction='mean')
        losses['E4'] = F.binary_cross_entropy(input=seg_edge_out4, target=edge_m, reduction='mean')
        return losses

########################################################################################################################