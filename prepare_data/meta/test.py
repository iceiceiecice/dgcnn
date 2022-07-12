import os
import sys
import glob
import h5py
import numpy as np
import torch
import json
import cv2
from torch.utils.data import Dataset

path = "C:\\Users\\xuzih\\Desktop\\test\\ply_data_train0.h5"
# path = 'C:\\dev\\dgcnn.pytorch\\data\\shapenet_part_seg_hdf5_data\\ply_data_train0.h5'
#
# train_dataset = h5py.File(path, 'r')
# train_set_x_orig = np.array(train_dataset['data'][:])  # your train set features
# # train_set_x2_orig = np.array(train_dataset['data_num'][:])  # your train set features
# train_set_y_orig = np.array(train_dataset['label'][:])  # your train set labels
# train_set_y2_orig = np.array(train_dataset['label_seg'][:])  # your train set labels
# # train_set_y2_orig = np.array(train_dataset['pid'][:])  # your train set labels
# # test_set_x_orig = np.array(train_dataset['X_test'][:])  # your train set features
# # test_set_y_orig = np.array(train_dataset['y_test'][:])  # your train set labels
# train_dataset.close()
# # 读写测试
# print(train_set_x_orig.shape)
# # print(train_set_x2_orig.shape)
# print(train_set_y_orig.shape)
# print(train_set_y2_orig.shape)
# #
# print(train_set_x_orig.max())
# print(train_set_x_orig.min())
# # print(train_set_x2_orig.max())
# # print(train_set_x2_orig.min())
# print(train_set_y_orig.max())
# print(train_set_y_orig.min())
# print(train_set_y2_orig.max())
# print(train_set_y2_orig.min())
# #
# # print(test_set_x_orig.shape)
# # print(test_set_y_orig.shape)


test = [1,2,3,4,5,6,7,8,9,10,11,12]

print(test[:100])