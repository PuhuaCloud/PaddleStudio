# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import os.path as osp
import shutil
from paddlex.utils import is_pic, get_encoding


class X2ImageNet(object):
    def __init__(self):
        pass

    def convert(self, image_dir, json_dir, dataset_save_dir):
        """转换。
        Args:
            image_dir (str): 图像文件存放的路径。
            json_dir (str): 与每张图像对应的json文件的存放路径。
            dataset_save_dir (str): 转换后数据集存放路径。
        """
        assert osp.exists(image_dir), "The image folder does not exist!"
        assert osp.exists(json_dir), "The json folder does not exist!"
        if not osp.exists(dataset_save_dir):
            os.makedirs(dataset_save_dir)
        assert len(os.listdir(
            dataset_save_dir)) == 0, "The save folder must be empty!"
        for img_name in os.listdir(image_dir):
            img_name_part = osp.splitext(img_name)[0]
            json_file = osp.join(json_dir, img_name_part + ".json")
            if not osp.exists(json_file):
                continue
            with open(
                    json_file, mode="r",
                    encoding=get_encoding(json_file)) as j:
                json_info = self.get_json_info(j)
                for output in json_info:
                    cls_name = output['name']
                    new_image_dir = osp.join(dataset_save_dir, cls_name)
                    if not osp.exists(new_image_dir):
                        os.makedirs(new_image_dir)
                    if is_pic(img_name):
                        shutil.copyfile(
                            osp.join(image_dir, img_name),
                            osp.join(new_image_dir, img_name))


class EasyData2ImageNet(X2ImageNet):
    """将使用EasyData标注的分类数据集转换为ImageNet数据集。
    """

    def __init__(self):
        super(EasyData2ImageNet, self).__init__()

    def get_json_info(self, json_file):
        json_info = json.load(json_file)
        json_info = json_info['labels']
        return json_info


class JingLing2ImageNet(X2ImageNet):
    """将使用标注精灵标注的分类数据集转换为ImageNet数据集。
    """

    def __init__(self):
        super(X2ImageNet, self).__init__()

    def get_json_info(self, json_file):
        json_info = json.load(json_file)
        json_info = json_info['outputs']['object']
        return json_info
