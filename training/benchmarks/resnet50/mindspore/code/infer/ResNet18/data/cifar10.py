#!/usr/bin/env python
# coding=utf-8

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import numpy as np

loc_1 = './train_cifar10/'
loc_2 = './test_cifar10/'

def unpickle(file_name):
    import pickle
    with open(file_name, 'rb') as fo:
        dict_res = pickle.load(fo, encoding='bytes')  # pylint: disable=too-many-arguments
    return dict_res

def cifar10_img():
    file_dir = './cifar-10-batches-py'
    os.mkdir(loc_1)
    os.mkdir(loc_2)
    for i in range(1, 6):
        data_name = os.path.join(file_dir, 'data_batch_' + str(i))
        data_dict = unpickle(data_name)
        print('{} is processing'.format(data_name))
        for j in range(10000):
            img = np.reshape(data_dict[b'data'][j], (3, 32, 32))
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_name = "%s%s%s.jpg" % (loc_1, str(data_dict[b'labels'][j]), str((i)*10000 + j))
            cv2.imwrite(img_name, img)
        print('{} is done'.format(data_name))
    test_data_name = file_dir + '/test_batch'
    print('{} is processing'.format(test_data_name))
    test_dict = unpickle(test_data_name)
    for m in range(10000):
        img = np.reshape(test_dict[b'data'][m], (3, 32, 32))
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_name = '%s%s%s%s' % (loc_2, str(test_dict[b'labels'][m]), str(10000 + m), '.png')
        img_label = "%s%s.png" % (str(test_dict[b'labels'][m]), str(10000 + m))
        cv2.imwrite(img_name, img)
        with open("test_label.txt", "a") as f:
            f.write(img_label + "\t" + str(test_dict[b'labels'][m]))
            f.write("\n")
    print("{} is done".format(test_data_name))
    print('Finish transforming to image')

if __name__ == '__main__':
    cifar10_img()
