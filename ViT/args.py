# -*- coding: UTF-8 -*-
import argparse
import os
import torch


def build_opt():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--traindir', type=str, default=None)
    # parser.add_argument('--train_metadir', type=str, default=None)
    # parser.add_argument('--train_metafile', type=str, default=None)

    # parser.add_argument('--testdir', type=str, default=None)
    # parser.add_argument('--test_metadir', type=str, default=None)
    # parser.add_argument('--test_metafile', type=str, default=None)


    parser.add_argument('--traindir', type=str, default='V:\\QiQiQiVVV\\QiMiracle\\VCreativeQi\\Dr.Qi\\SuccessQi\\V\HUAWEI\\AnomalyDetectionCode\\Classification\\labeled_data_test') #V:\QiQiQiVVV\QiMiracle\VCreativeQi\Dr.Qi\SuccessQi\V\HUAWEI\AnomalyDetectionCode\Classification\labeled_data_test
    parser.add_argument('--train_metadir', type=str, default='V:\\QiQiQiVVV\\QiMiracle\\VCreativeQi\\Dr.Qi\\SuccessQi\\V\HUAWEI\\AnomalyDetectionCode\\Classification')
    parser.add_argument('--train_metafile', type=str, default='train_label_test.csv')

    # parser.add_argument('--traindir', type=str, default='E:\\2022_2_data\\train_image\\train_image\\labeled_data_test') #
    # parser.add_argument('--train_metadir', type=str, default='E:\\2022_2_data\\train_image\\train_image')
    # parser.add_argument('--train_metafile', type=str, default='train_label_test.csv')
    
    
    # parser.add_argument('--testdir', type=str, default='V:\\QiQiQiVVV\\QiMiracle\\VCreativeQi\\Dr.Qi\\SuccessQi\\V\HUAWEI\\AnomalyDetectionCode\\Classification\\labeled_data_test')
    # parser.add_argument('--test_metadir', type=str, default='V:\\QiQiQiVVV\\QiMiracle\\VCreativeQi\\Dr.Qi\\SuccessQi\\V\HUAWEI\\AnomalyDetectionCode\\Classification')
    # parser.add_argument('--test_metafile', type=str, default='train_label_test.csv')
    
    parser.add_argument('--testdir', type=str, default='V:\\QiQiQiVVV\\QiMiracle\\VCreativeQi\\Dr.Qi\\SuccessQi\\V\HUAWEI\\2022_2_data\\test_images')
    #parser.add_argument('--test_metadir', type=str, default='V:\\QiQiQiVVV\\QiMiracle\\VCreativeQi\\Dr.Qi\\SuccessQi\\V\HUAWEI\\AnomalyDetectionCode\\Classification')
    parser.add_argument('--test_metadir', type=str, default= 'V:\\QiQiQiVVV\\QiMiracle\\VCreativeQi\\Dr.Qi\\SuccessQi\\V\HUAWEI\\2022_2_data')
    parser.add_argument('--test_metafile', type=str, default='test.csv') #None default='train_label_test.csv'
    
    
    
    parser.add_argument('--epochs', type=int, default=3) #9
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1) #128 original  16 not work 8 ok
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=int, default=0.9)

    opt = parser.parse_args()

    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    opt.resultdir = './result'
    if not os.path.exists(opt.resultdir):
        os.makedirs(opt.resultdir)

    print('.' * 75)
    for key in opt.__dict__:
        param = opt.__dict__[key]
        print('...param: {}: {}'.format(key, param))
    print('.' * 75)

    return opt


