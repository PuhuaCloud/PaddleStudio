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

from __future__ import absolute_import
import copy
import os.path as osp
import six
import sys
import numpy as np
from paddlex.utils import logging, is_pic, get_num_workers
from .voc import VOCDetection
from paddlex.cv.transforms import MixupImage


class CocoDetection(VOCDetection):
    """读取MSCOCO格式的检测数据集，并对样本进行相应的处理，该格式的数据集同样可以应用到实例分割模型的训练中。

    Args:
        data_dir (str): 数据集所在的目录路径。
        ann_file (str): 数据集的标注文件，为一个独立的json格式文件。
        transforms (paddlex.det.transforms): 数据集中每个样本的预处理/增强算子。
        num_workers (int|str): 数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据
            系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。
        shuffle (bool): 是否需要对数据集中样本打乱顺序。默认为False。
        allow_empty (bool): 是否加载负样本。默认为False。
        empty_ratio (float): 用于指定负样本占总样本数的比例。如果小于0或大于等于1，则保留全部的负样本。默认为1。
    """

    def __init__(self,
                 data_dir,
                 ann_file,
                 transforms=None,
                 num_workers='auto',
                 shuffle=False,
                 allow_empty=False,
                 empty_ratio=1.):
        # matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
        # or matplotlib.backends is imported for the first time
        # pycocotools import matplotlib
        import matplotlib
        matplotlib.use('Agg')
        from pycocotools.coco import COCO

        try:
            import shapely.ops
            from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
        except:
            six.reraise(*sys.exc_info())

        super(VOCDetection, self).__init__()
        self.data_dir = data_dir
        self.data_fields = None
        self.transforms = copy.deepcopy(transforms)
        self.num_max_boxes = 50
        self.use_mix = False
        if self.transforms is not None:
            for op in self.transforms.transforms:
                if isinstance(op, MixupImage):
                    self.mixup_op = copy.deepcopy(op)
                    self.use_mix = True
                    self.num_max_boxes *= 2
                    break

        self.batch_transforms = None
        self.num_workers = get_num_workers(num_workers)
        self.shuffle = shuffle
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio
        self.file_list = list()
        neg_file_list = list()
        self.labels = list()

        coco = COCO(ann_file)
        self.coco_gt = coco
        img_ids = sorted(coco.getImgIds())
        cat_ids = coco.getCatIds()
        catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        cname2clsid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in catid2clsid.items()
        })
        for label, cid in sorted(cname2clsid.items(), key=lambda d: d[1]):
            self.labels.append(label)
        logging.info("Starting to read file list from dataset...")

        ct = 0
        for img_id in img_ids:
            is_empty = False
            img_anno = coco.loadImgs(img_id)[0]
            im_fname = osp.join(data_dir, img_anno['file_name'])
            if not is_pic(im_fname):
                continue
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])
            ins_anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
            instances = coco.loadAnns(ins_anno_ids)

            bboxes = []
            for inst in instances:
                x, y, box_w, box_h = inst['bbox']
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(im_w - 1, x1 + max(0, box_w))
                y2 = min(im_h - 1, y1 + max(0, box_h))
                if inst['area'] > 0 and x2 >= x1 and y2 >= y1:
                    inst['clean_bbox'] = [x1, y1, x2, y2]
                    bboxes.append(inst)
                else:
                    logging.warning(
                        "Found an invalid bbox in annotations: "
                        "im_id: {}, area: {} x1: {}, y1: {}, x2: {}, y2: {}."
                        .format(img_id, float(inst['area']), x1, y1, x2, y2))
            num_bbox = len(bboxes)
            if num_bbox == 0 and not self.allow_empty:
                continue
            elif num_bbox == 0:
                is_empty = True

            gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
            gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
            gt_score = np.ones((num_bbox, 1), dtype=np.float32)
            is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
            difficult = np.zeros((num_bbox, 1), dtype=np.int32)
            gt_poly = [None] * num_bbox

            has_segmentation = False
            for i, box in reversed(list(enumerate(bboxes))):
                catid = box['category_id']
                gt_class[i][0] = catid2clsid[catid]
                gt_bbox[i, :] = box['clean_bbox']
                is_crowd[i][0] = box['iscrowd']
                if 'segmentation' in box and box['iscrowd'] == 1:
                    gt_poly[i] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
                elif 'segmentation' in box and box['segmentation']:
                    if not np.array(box[
                            'segmentation']).size > 0 and not self.allow_empty:
                        gt_poly.pop(i)
                        is_crowd = np.delete(is_crowd, i)
                        gt_class = np.delete(gt_class, i)
                        gt_bbox = np.delete(gt_bbox, i)
                    else:
                        gt_poly[i] = box['segmentation']
                    has_segmentation = True
            if has_segmentation and not any(gt_poly) and not self.allow_empty:
                continue

            im_info = {
                'im_id': np.array([img_id]).astype('int32'),
                'image_shape': np.array([im_h, im_w]).astype('int32'),
            }
            label_info = {
                'is_crowd': is_crowd,
                'gt_class': gt_class,
                'gt_bbox': gt_bbox,
                'gt_score': gt_score,
                'gt_poly': gt_poly,
                'difficult': difficult
            }

            if is_empty:
                neg_file_list.append({
                    'image': im_fname,
                    **
                    im_info,
                    **
                    label_info
                })
            else:
                self.file_list.append({
                    'image': im_fname,
                    **
                    im_info,
                    **
                    label_info
                })
            ct += 1

            if self.use_mix:
                self.num_max_boxes = max(self.num_max_boxes,
                                         2 * len(instances))
            else:
                self.num_max_boxes = max(self.num_max_boxes, len(instances))

        if not ct:
            logging.error(
                "No coco record found in %s' % (ann_file)", exit=True)
        self.pos_num = len(self.file_list)
        if self.allow_empty and neg_file_list:
            self.file_list += self._sample_empty(neg_file_list)
        logging.info(
            "{} samples in file {}, including {} positive samples and {} negative samples.".
            format(
                len(self.file_list), ann_file, self.pos_num,
                len(self.file_list) - self.pos_num))
        self.num_samples = len(self.file_list)

        self._epoch = 0
