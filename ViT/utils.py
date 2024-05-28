import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
#import config
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as F1
#from IPython.display import display


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) #
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
class FocalLoss_Mult(nn.Module):
    def __init__(self,alpha,gamma,size_average=True):#alpha=0.25,gamma=2.0,epsilon=1.0
        super(FocalLoss_Mult,self).__init__()
        self.alpha=torch.tensor(alpha).cuda()
        self.gamma=gamma
        #self.epsilon=epsilon
        # self.num_classes=num_classes
        self.size_average = size_average

    def forward(self, pred, target):
        criterion = FocalLoss(self.alpha,self.gamma,self.size_average)
        loss = torch.zeros(1,target.shape[1]).cuda()
        
        for label in range(target.shape[1]):
            batch_loss = criterion(pred[:,label],target[:,label])
            loss[0,label] = batch_loss.mean()

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
class FocalLoss(nn.Module):
    def __init__(self,alpha,gamma,num_classes=2,size_average=True):#alpha=0.25,gamma=2.0,epsilon=1.0
        super(FocalLoss,self).__init__()
        self.alpha=torch.tensor(alpha).cuda()
        self.gamma=gamma
        #self.epsilon=epsilon
        self.num_classes=num_classes
        self.size_average = size_average

    def forward(self, pred1, target):
        pred = nn.Sigmoid()(pred1)
        p=pred
        target1 = F.one_hot(target.unsqueeze(1), self.num_classes).transpose(1, -1).squeeze_(-1)
        target1 = target1.to(device=pred1.device,dtype=pred1.dtype)
        pt=target1*p +(1-target1)* (1-p)     
        p = p.view(-1,1)
        pred = pred.view(-1,1)
        pt = pt.contiguous().view(-1,1)
        target = target.view(-1,1)
        
        pred = torch.cat((pred,1-pred),dim=1)
        class_mask = torch.zeros(pred.shape[0],pred.shape[1]).cuda()
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)
        probs = (pred * class_mask).sum(dim=1).view(-1,1)
        probs = probs.clamp(min=0.0001,max=1.0)     
        
        log_p = pt.log()
        # alpha = torch.ones(pred.shape[0],pred.shape[1]).cuda()
        # alpha[:,0] = alpha[:,0] * self.alpha
        # alpha[:,1] = alpha[:,1] * (1-self.alpha)
        # alpha = (alpha * class_mask).sum(dim=1).view(-1,1)

        FL = -self.alpha*(torch.pow((1-pt), self.gamma))*log_p
        #poly=FL+self.epsilon*torch.pow(1-pt,self.gamma+1)

        if self.size_average:
            loss = FL.mean()
        else:
            loss = FL.sum()
        return loss
class FocalLoss_poly(nn.Module):
    def __init__(self,alpha,gamma,epsilon,num_classes=2,size_average=True):#alpha=0.25,gamma=2.0,epsilon=1.0
        super(FocalLoss_poly,self).__init__()
        self.alpha=torch.tensor(alpha).cuda()
        self.gamma=gamma
        self.epsilon=epsilon
        self.num_classes=num_classes
        self.size_average = size_average

    def forward(self, pred1, target):
        pred = nn.Sigmoid()(pred1)
        p=pred
        target1 = F.one_hot(target.unsqueeze(1), self.num_classes).transpose(1, -1).squeeze_(-1)
        target1 = target1.to(device=pred1.device,dtype=pred1.dtype)
        pt=target1*p +(1-target1)* (1-p)     
        p = p.view(-1,1)
        pred = pred.view(-1,1)
        pt = pt.contiguous().view(-1,1)
        target = target.view(-1,1)
        
        pred = torch.cat((pred,1-pred),dim=1)
        class_mask = torch.zeros(pred.shape[0],pred.shape[1]).cuda()
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)
        probs = (pred * class_mask).sum(dim=1).view(-1,1)
        probs = probs.clamp(min=0.0001,max=1.0)     
        #pt=pt.view(-1,1)
        #pt = (pt * class_mask).sum(dim=1).view(-1,1)
        #pt = pt.clamp(min=0.0001,max=1.0)
        
        log_p = probs.log()
        alpha = torch.ones(pred.shape[0],pred.shape[1]).cuda()
        alpha[:,0] = alpha[:,0] * self.alpha
        alpha[:,1] = alpha[:,1] * (1-self.alpha)
        alpha = (alpha * class_mask).sum(dim=1).view(-1,1)

        FL = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        poly=FL+self.epsilon*torch.pow(1-pt,self.gamma+1)

        if self.size_average:
            loss = poly.mean()
        else:
            loss = poly.sum()
        return loss
# class poly_FocalLoss(nn.Module):
    
#     def __init__(self, epsilon=1.0, gamma=2.0):
#         super(poly_FocalLoss, self).__init__()
#         self.epsilon = epsilon
#         self.gamma = gamma
        
#     def forward(self, p, pt):
#         p=torch.sigmoid(p)
#         pt=pt*p+(1-pt)*(1-p)
#         FL=FocalLoss(pt,self.gamma)
#         # print(p.shape)
#         # print(pt.shape)
#         # print(FL)
#         polyFL=torch.add(FL,self.epsilon*torch.pow(1-pt, self.gamma+1))
        
#         return polyFL
'''
class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    
    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
'''

class Poly1CrossEntropyLoss(nn.Module):
    def __init__(self,
                 num_classes: int=2,
                 epsilon: float = 0.001,
                 reduction: str = "mean"):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(device=logits.device, dtype=logits.dtype)
        #print(labels_onehot.shape)#torch.Size([2, 128, 256, 2])
        #print(logits.shape)#torch.Size([2, 2, 128, 256])
        labels_onehot = torch.einsum('nhwc->nchw', labels_onehot)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=1)
        CE = F.cross_entropy(input=logits, target=labels, reduction='none')
        #print('=============')
        #print(CE.shape)#torch.Size([2, 128, 256])
       # print(pt.shape)#torch.Size([2, 2, 128])
        poly1 = CE + self.epsilon * (1 - pt)
        #if self.reduction == "mean":
        poly1 = poly1.mean()
       # elif self.reduction == "sum":
        #    poly1 = poly1.sum()
        return poly1



#https://github.com/abhuse/polyloss-pytorch/blob/main/polyloss.py
class Poly1FocalLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "none"):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        """
        super(Poly1FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth of shape [N] or [N, ...], NOT one-hot encoded
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py

        p = torch.sigmoid(logits)
        print(labels.shape)
        # if labels are of shape [N]
        # convert to one-hot tensor of shape [N, num_classes]
        if labels.ndim == 1:
            labels = F.one_hot(labels, num_classes=self.num_classes)

        # if labels are of shape [N, ...] e.g. segmentation task
        # convert to one-hot tensor of shape [N, num_classes, ...]
        else:
            labels = F.one_hot(labels.unsqueeze(1), self.num_classes).transpose(1, -1).squeeze_(-1)

        labels = labels.to(device=logits.device,
                           dtype=logits.dtype)
        print(labels.shape)
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1

#https://zhuanlan.zhihu.com/p/75542467

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
      
        pred = nn.Sigmoid()(pred)
       
        pred = pred.view(-1,1)
        target = target.view(-1,1)
      
        pred = torch.cat((pred,1-pred),dim=1)
    
        class_mask = torch.zeros(pred.shape[0],pred.shape[1]).cuda()
    
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        probs = (pred * class_mask).sum(dim=1).view(-1,1)
        probs = probs.clamp(min=0.0001,max=1.0)

        log_p = probs.log()


        alpha = torch.ones(pred.shape[0],pred.shape[1]).cuda()
        # alpha[:,0] = alpha[:,0] * (1-self.alpha)
        # alpha[:,1] = alpha[:,1] * self.alpha
        alpha[:,0] = alpha[:,0] * (1-self.alpha)
        alpha[:,1] = alpha[:,1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1,1)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss
#https://github.com/HeyLynne/FocalLoss_for_multiclass/blob/master/loss_helper.py
class FocalLoss1(nn.Module):
    def __init__(self, gamma = 2, alpha = 1, size_average = True):
        super(FocalLoss1, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert(logits.size(0) == labels.size(0))
        assert(logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1).to(device)
        label_onehot = torch.zeros([batch_size, labels_length, seq_length]).scatter_(1, new_label, 1).to(device)
        # label_onehot = label_onehot.permute(0, 2, 1) # transpose, batch_size * seq_length * labels_length

        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()
#https://github.com/CoinCheung/pytorch-loss
## version 1: use torch.autograd

class FocalLoss2(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''

        label = F.one_hot(label.unsqueeze(1), 2).transpose(1, -1).squeeze_(-1)
        #label = torch.einsum('nhwc->nchw', label)
        probs = nn.Sigmoid()(logits)#torch.sigmoid()
        # print(label.shape)#torch.Size([5, 128, 256])  
        # print(probs.shape)#torch.Size([5, 2, 128, 256])
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha *log_probs  + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss






    
'''https://blog.csdn.net/weixin_39745724/article/details/110645619
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)
 
    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore
            labels = labels[mask]
            preds = preds[mask]
 
        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
''' 
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        #print(x1.shape)
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        #print(diffX)
        diffY = x1.size()[3] - x2.size()[3]
        #print(diffY)
        #print(x2.shape)
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        #print(x1.shape)
        #print(x2.shape)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        x = self.conv(x)
        return x


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda(),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda())


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor=input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])


                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output = layer_output.permute(1, 0, 2, 3, 4)

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding """
#     def __init__(self, img_size=(config.img_height, config.img_width), patch_size=(16, 16), in_chans=3, embed_dim=768):
#         super().__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = 450#(img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
#         self.patch_shape = (16,16)#(img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x, **kwargs):
#         B, C, H, W = x.shape
#         assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         return x 


# Down sampling module
def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.ReLU(),
        nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.ReLU()
    )

# Up sampling module
def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ReLU()
    )

