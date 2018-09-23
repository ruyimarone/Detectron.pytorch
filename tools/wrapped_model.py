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
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset', required=True,
        help='training dataset')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--image_dir',
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        help='images to infer. Must not use with --image_dir')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="infer_outputs")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=True)

    args = parser.parse_args()

    return args

class WrappedDetectron:
    def __init__(self):
        if not torch.cuda.is_available():
            sys.exit("Need a CUDA device to run the code.")

        args = Namespace(dataset = 'coco',
                cfg_file='configs/baselines/e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml',
                load_detectron='data/model_final.101.pkl',
                image_dir='test_imgs/',
                cuda=True)

        print('Called with args:')
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

        maskRCNN = Generalized_RCNN()

        if args.cuda:
            maskRCNN.cuda()

        print("loading detectron weights %s" % args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

        maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                     minibatch=True, device_ids=[0])  # only support single GPU

        maskRCNN.eval()
        self.model = maskRCNN
        print("loaded models")

    def forward(self, filename):
        im = cv2.imread(filename)
        cls_boxes, cls_segms, cls_keyps = im_detect_all(self.model, im)
        return cls_boxes, cls_segms, cls_keyps

