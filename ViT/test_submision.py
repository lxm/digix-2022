# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 04:30:54 2022

@author: LocalAdmin
"""

# -*- coding: UTF-8 -*-
import os
import torch
import torchvision
import time
import csv
from args import build_opt
from loaders import build_loader

import torch.optim as optim
import torch.nn as nn

from timm.models.vision_transformer import vit_large_patch16_224_in21k as create_model  # vit_base_patch16_224_in21k, vit_large_patch16_224_in21k  ;vit_huge_patch14_224_in21k # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py


opt = build_opt()
#model = torchvision.models.mobilenet_v3_large(pretrained=False) #model = torchvision.models.mobilenet_v3_large(pretrained=True, num_classes=opt.num_classes)

model = create_model(num_classes=2)
# model = torchvision.models.mobilenet_v3_large(pretrained=True) #model = torchvision.models.mobilenet_v3_large(pretrained=True, num_classes=opt.num_classes)
# num_ftrs = model.classifier[3].in_features
# model.classifier[3] = nn.Linear(num_ftrs, 2)
model.cuda() #Add for parallel
model = nn.DataParallel(model)
model.to(opt.device)

# pretrained_dict = torch.load('./checkpoints/epoch31_valloss0.1407071848710378.pth')
# model_dict = model.state_dict()
# pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
# model_dict.update(pretrained_dict_1)
# model.load_state_dict(model_dict)

checkpoint = torch.load('./checkpoints/epoch215_valloss0.22969300413056265.pth')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
to_prob = nn.Softmax(dim=1)

test_loader = build_loader(
    imagedir=opt.testdir,
    batch_size=opt.batch_size,
    num_workers=opt.num_workers,
    metadir=opt.test_metadir,
    metafile=opt.test_metafile,
    require_label=False
)

with torch.no_grad():
    imagenames, probs = list(), list()
    for batch_idx, batch in enumerate(test_loader):
        image, imagename = batch
        image = image.to(opt.device)
        pred = model(image)
        prob = to_prob(pred)
        res = prob.data.cpu().numpy()
        prob = list(res)
        imagenames += imagename
        print(imagename)
        probs += prob

with open(os.path.join(opt.resultdir, 'submission.csv'), 'w', encoding='utf8') as fp:
    writer = csv.writer(fp,dialect='unix',quoting=csv.QUOTE_NONE, quotechar='')
    writer.writerow(['imagename', 'defect_prob'])
    for info in zip(imagenames, probs):
        imagename, prob = info
        writer.writerow([imagename, str(prob[1])])
            
# with open(os.path.join(opt.resultdir, 'submission.csv'), 'w', encoding='utf8') as fp:
#     writer = csv.writer(fp)
#     writer.writerow(['imagename', 'defect_prob'])
#     for info in zip(imagenames, probs):
#         imagename, prob = info
#         res_float = float('{:.10f}'.format(prob[1]))
#         res_final = res_float
#         if res_float < 0.1:
#             res_final = 0
#         if res_float > 0.99:
#             res_final = 1
#         writer.writerow([imagename, str(res_final)])
