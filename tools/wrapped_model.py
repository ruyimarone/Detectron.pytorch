from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import pprint
import subprocess
from collections import defaultdict
from six.moves import xrange
from argparse import Namespace

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable

import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_get_all_masks
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

from functools import reduce

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def union_masks(masks):
    if len(masks) == 1:
        return masks[0] == 1
    return reduce(lambda a, b : (a == 1) | (b == 1), masks)

# def interpolate_mask(a, b, mask):
    #combine a and b according to the weights in mask ([0, 1])
    #broadcast the mask to 3 channels, if needed
    # if len(a.shape) == 3:
        # new_mask = np.zeros(a.shape)
        # new_mask[:, :, 0] = mask
        # new_mask[:, :, 1] = mask
        # new_mask[:, :, 2] = mask
        # mask = new_mask
    # return (a * (1.0 - mask)) + (b * (mask))

def grayscale(im):
    single_channel = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2GRAY)
    out = np.zeros(im.shape)
    out[:, :, 0] = single_channel.copy()
    out[:, :, 1] = single_channel.copy()
    out[:, :, 2] = single_channel.copy()
    return out.astype('uint8')

def apply_binary_mask(source, content, mask):
    s = source.copy()
    s[mask] = 0
    c = content.copy()
    c[~mask] = 0
    return s + c, s, c


class WrappedDetectron:
    def __init__(self):

        ##########
        # CONFIG #
        ##########

        #backbone models are effectively singletons
        #so we can only run the config code once

        try:
            if not torch.cuda.is_available():
                sys.exit("Need a CUDA device to run the code.")

            args = Namespace(dataset = 'coco',
                    cfg_file='configs/baselines/e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml',
                    load_detectron='data/model_final.101.pkl',
                    image_dir='test_imgs/',
                    cuda=True)

            print("Loaded with parameters:")
            print(args)

            assert args.image_dir or args.images
            assert bool(args.image_dir)

            if args.dataset.startswith("coco"):
                dataset = datasets.get_coco_dataset()
                cfg.MODEL.NUM_CLASSES = len(dataset.classes)
            elif args.dataset.startswith("keypoints_coco"):
                dataset = datasets.get_coco_dataset()
                cfg.MODEL.NUM_CLASSES = 2
            else:
                raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

            print('load cfg from file: {}'.format(args.cfg_file))
            cfg_from_file(args.cfg_file)

            assert bool(args.load_detectron)
            cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
            assert_and_infer_cfg()
        except AttributeError:
            print("already loaded!")
            pass


        maskRCNN = Generalized_RCNN()

        if args.cuda:
            maskRCNN.cuda()

        print("loading detectron weights %s" % args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

        maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                     minibatch=True, device_ids=[0])  # only support single GPU

        maskRCNN.eval()
        self.model = maskRCNN
        self.dataset = dataset
        print("loaded models")

    def segment_people(self, filename, threshold=0.9):
        person_id = 1
        cls_boxes, cls_segms, masks = self.forward(filename)
        person_boxes = cls_boxes[person_id]
        person_masks = masks[person_id]
        #bounding boxes are returned as corners followed by score
        return [(mask, bbox[-1]) for mask, bbox in zip(person_masks, person_boxes) if bbox[-1] > 0.9]

    def forward(self, filename):
        im = cv2.imread(filename)
        cls_boxes, cls_segms, masks = im_get_all_masks(self.model, im)
        return cls_boxes, cls_segms, masks

