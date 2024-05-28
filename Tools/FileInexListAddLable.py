# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 19:01:58 2022

@author: 29499
"""

import os
import re
import csv
from itertools import islice
images_path = '/tudelft.net/staff-umbrella/SAMEN/DeepLearning/HUAWEI/2022_2_data/train_image_crop1080/labeled_data'   # 图片存放目录
txt_save_path = '/tudelft.net/staff-umbrella/SAMEN/DeepLearning/HUAWEI/2022_2_data/Train_ImageIndexwith_label.txt'  # 生成的图片列表清单txt文件名
fw = open(txt_save_path, "w")
# 读取函数，用来读取文件夹中的所有函数，输入参数是文件名
def read_directory(directory_name):
	images = list()
	labels = list()
	with open(os.path.join('/tudelft.net/staff-umbrella/SAMEN/DeepLearning/HUAWEI/2022_2_data/train_label','train_label.csv'), 'r', encoding='utf8') as fp:
		reader = csv.reader(fp)
		for row in islice(reader, 0, None):
			imagename = row[0]
			images.append(imagename)
			class_label = row[1]
			if class_label == '0':
				label = 0
					# print('0')
			else:
				label = 1
					# print('1')
			labels.append(label)
	name=imagename
	label=labels
	i=0
	for filename in os.listdir(directory_name):
		print(filename)  # 仅仅是为了测试
# 		print(name)
# 		print(label[i])
		fw.write(images_path + filename + ' ')  # 打印成功信息
		fw.write(images_path + filename + ' ')  # 打印成功信息
		fw.write(images_path + filename + ' ')  # 打印成功信息
		fw.write(images_path + filename + ' ')  # 打印成功信息
		fw.write(images_path + filename + ' ')  # 打印成功信息
		

# 		fw.write('E:\\2022_2_data\\train_image\\train_image\\labeled_data\\' + filename + '\n')  # 打印成功信息

		fw.write( str(label[i]) + '\n')  # 打印成功信息
		i+=1
		# img = cv2.imread(directory_name + "/" + filename)
		# #####显示图片#######
		# cv2.imshow(filename, img)
		# cv2.waitKey(0)
		# #####################
		#
		# #####保存图片#########
		# cv2.imwrite("D://wangyang//face1" + "/" + filename, img)


read_directory(images_path)#这里传入所要读取文件夹的绝对路径，加引号（引号不能省略！）

# def get_labels(self, metadir=config.train_metadir, metafile=config.train_metafile):
# 		images = list()
# 		labels = list()
# 		with open(os.path.join(metadir, metafile), 'r', encoding='utf8') as fp:
# 			reader = csv.reader(fp)
# 			for row in islice(reader, 0, None):
# 				imagename = row[0]
# 				images.append(imagename)
# 				class_label = row[1]
# 				if class_label == '0':
# 					label = 0
# 					# print('0')
# 				else:
# 					label = 1
# 					# print('1')
# 				labels.append(label)
# 		return imagename,labels