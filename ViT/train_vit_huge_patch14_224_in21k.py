# -*- coding: UTF-8 -*-
import os
import torch
import torchvision
import time
import csv

import torch.optim as optim
import torch.nn as nn
import tensorflow as tf
import torch.nn.functional as F


from args import build_opt
from loaders import build_loader
from utils import Poly1FocalLoss,FocalLoss,FocalLoss_poly,FocalLoss1,FocalLoss_Mult,BCEFocalLoss
from radam import RAdam

from timm.models.vision_transformer import vit_huge_patch14_224_in21k as create_model  # vit_base_patch16_224_in21k, vit_large_patch16_224_in21k  ;vit_huge_patch14_224_in21k # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# test vit_huge_patch14_224_in21k by simple replace the name above should work
def poly1_cross_entropy(epsilon=1.0):
    def _poly1_cross_entropy(y_pred,y_true):
        # pt, CE, and Poly1 have shape [batch].
        pt = tf.reduce_sum(y_true * tf.nn.softmax(y_pred), axis=-1)
        CE = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
        Poly1 = CE + epsilon * (1 - pt)
        loss = tf.reduce_mean(Poly1)
        return loss
    return _poly1_cross_entropy

class Poly1CrossEntropyLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 reduction: str = "none"):
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
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        CE = F.cross_entropy(input=logits, target=labels, reduction='none')
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1

class PolyLoss(torch.nn.Module):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    """

    def __init__(self, epsilon=2.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        ce = F.cross_entropy(outputs, targets)
        pt = F.one_hot(targets, outputs.size()[1]) * F.softmax(outputs, 1)
        return (ce + self.epsilon * (1.0 - pt.sum(dim=1))).mean()
    
def main():
    opt = build_opt()
    # model = torchvision.models.mobilenet_v3_large(pretrained=True) #model = torchvision.models.mobilenet_v3_large(pretrained=True, num_classes=opt.num_classes)
    # num_ftrs = model.classifier[3].in_features
    # model.classifier[3] = nn.Linear(num_ftrs, 2)
    model = create_model(num_classes=2) # model = create_model(num_classes=2, has_logits=False)
    model.cuda() #Add for parallel
    model = nn.DataParallel(model)
    model.to(opt.device)
    # optimizer = RAdam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=opt.weight_decay) #Currently not perform well
    #optimizer = RAdam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999)) #Currently not perform well
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay) #not well
    #optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr) #not well
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.lr) #not well

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay) # , weight_decay=opt.weight_decay without not good
    # loss = nn.CrossEntropyLoss() #
    #loss =nn.BCELoss()
    loss = PolyLoss() #
    
    
    train_loader = build_loader(
        imagedir=opt.traindir,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        metadir=opt.train_metadir,
        metafile=opt.train_metafile,
        require_label=True
    )
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)

    test_loader = build_loader(
        imagedir=opt.testdir,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        metadir=opt.test_metadir,
        metafile=opt.test_metafile,
        require_label=False
    )

    print('...start training')

    for epoch in range(opt.epochs):
        start_t = time.time()
        epoch_l = 0
        epoch_t = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            image, label = batch
            image, label = image.to(opt.device), label.to(opt.device)
            output = model(image)

            l = loss(output, label)
            l.backward()
            optimizer.step()

            batch_l = l.item()
            epoch_l += batch_l
            batch_t = time.time() - start_t
            epoch_t += batch_t
            start_t = time.time()
        lr_scheduler.step()
        epoch_t = epoch_t / len(train_loader)
        epoch_l = epoch_l / len(train_loader)
        torch.save(
            {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_l,
            },
            './checkpoints/epoch{}_valloss{}.pth'.format(epoch+1,epoch_l))
        
        #torch.save(model.state_dict(), './model_pretrain/epoch{}_valloss{}_{}_{}.pth'.format(epoch,test_loss,args.optimizer,args.lr))

        
        print('...epoch: {:3d}/{:3d}, loss: {:.4f}, average time: {:.2f}.'.format(
            epoch + 1, opt.epochs, epoch_l, epoch_t))
        print('\n lr:')
        print(lr_scheduler.get_last_lr())


    print('V You can start predicting V.......')
    
'''
    model.eval()
    to_prob = nn.Softmax(dim=1)
    with torch.no_grad():
        imagenames, probs = list(), list()
        for batch_idx, batch in enumerate(test_loader):
            image, imagename = batch
            image = image.to(opt.device)
            pred = model(image)
            prob = to_prob(pred)
            prob = list(prob.data.cpu().numpy())
            imagenames += imagename
            probs += prob

    with open(os.path.join(opt.resultdir, 'submission.csv'), 'w', encoding='utf8') as fp:
        writer = csv.writer(fp,dialect='unix',quoting=csv.QUOTE_NONE, quotechar='')
        writer.writerow(['imagename', 'defect_prob'])
        for info in zip(imagenames, probs):
            imagename, prob = info
            writer.writerow([imagename, str(prob[1])])
'''

    
if __name__ == "__main__":
    print('.' * 75)
    main()
    print('.' * 75)




