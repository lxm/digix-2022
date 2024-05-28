# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 22:46:28 2022

@author: DYQ
"""

import os
#import re
#f_path = '/tudelft.net/staff-umbrella/SAMEN/DeepLearning/HUAWEI/2022_2_data/train_image/labeled_data/'   #File Dir  \2022_2_data\train_image
#f_path = '/tudelft.net/staff-umbrella/SAMEN/DeepLearning/HUAWEI/2022_2_data/test_images/'
f_path = '/tudelft.net/staff-umbrella/SAMEN/DeepLearning/HUAWEI/2022_2_data/train_image/unlabeled_data/'
#txt_save_path = './train_labeled_index.txt'  # Index save file txt
txt_save_path = './train_unlabeled_index.txt'  # Index save file txt
#txt_save_path = './text_index.txt'  # Index save file txt


fw = open(txt_save_path, "w")
# 读取函数，用来读取文件夹中的所有函数，输入参数是文件名
def read_directory(directory_name):
	for filename in os.listdir(directory_name):
		print(filename)  # just for test
		filepath = os.path.join(f_path, filename)        
		fw.write(filepath + ' ')
		fw.write(filepath + ' ')
		fw.write(filepath + ' ')
		fw.write(filepath + ' ')
		fw.write(filepath + ' ')        
		fw.write(filepath + '\n')  # 打印成功信息 + ' '
		# img = cv2.imread(directory_name + "/" + filename)
		# #####show image#######
		# cv2.imshow(filename, img)
		# cv2.waitKey(0)
		# #####################
		#
		# #####save image#########
		# cv2.imwrite("D://wangyang//face1" + "/" + filename, img)


read_directory(f_path)#