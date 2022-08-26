# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os, random, shutil
import time
import cv2
import tqdm
import numpy as np
import copy
import torch
import sys
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_setup
#from detectron2.data.detection_utils import convert_image_to_rgb
import torchvision.transforms.functional as F
sys.path.append('/home/lzr/projects/projects/Entity-main/Entity/EntitySeg')
from entityseg import *
from PIL import Image

from predictor import VisualizationDemo
from collections import defaultdict


# constants

WINDOW_NAME = "Image Segmentation"

'''
def combine_img(pil_img, mask, save_path, num):
    img = np.array(pil_img)
    img = torch.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 255
    print(img.shape)

    # mask = mask
    # mask = (mask > 0).astype(np.uint8) * 255 
    # mask = to_tensor(mask)
    # print(mask.shape)

    #source_mask_64 = torch.ones_like(img)[:, :1]
    #source_mask_64[:, :, 20:] = 0
    #print(source_mask_64.shape)
    #source_mask_256 = F.interpolate(source_mask_64, (256, 256), mode='nearest')

    res = mask * img
    
    y = res.permute(2, 0, 3, 1)
    Image.fromarray(y.numpy()).save(save_path + 'res_{}.jpg'.format(''.join(str(num))))
'''

def generate_mask(name, image, color, num, save_path):
    img = image
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    #mask = torch.ones_like(source_image_64)[:, :1]
    height = img.shape[0]
    weight = img.shape[1]
    channels = img.shape[2]
    for row in range(height):
        for col in range(weight):
            r=img[row, col, 0]
            g=img[row, col, 1]
            b=img[row, col, 2]
            
            if([r,g,b]==color):
                mask[row][col]=255
                # continue
            else:
                mask[row][col]=0
    # print(mask.shape)
    # print(type(mask))
    # print(img.shape)
    # print(type(img))
    # print(mask.shape)
    # print(type(mask))
    
    # mask_mul = torchvision.transforms.functional.to_tensor(mask)
    # img_mul = torchvision.transforms.functional.to_tensor(img)

    
    # res = (mask_mul*img_mul).permute(1, 2, 0).numpy()
    
    #mask = to_tensor(mask)
    #image_for_mul = to_tensor(ori_img)
    #result = image_for_mul * mask  
    #y = result.permute(2, 0, 3, 1).reshape([(result).shape[2], -1, 3])
    #Image.fromarray(result.numpy()).save('/home/lzr/projects/Entity-main/Entity/EntitySeg/mask_output/1.jpg')
    #path_res = '/home/lzr/projects/Entity-main/Entity/EntitySeg/mask_output/'+'res_{}.jpg'.format(''.join(str(num)))
    path_mask = save_path + '/' +name + '_mask_{}.jpg'.format(''.join(str(num)))
    
    #Image.fromarray((mask*img).numpy()).save('/home/lzr/projects/Entity-main/Entity/EntitySeg/mask_output/'+'mul_{}.jpg'.format(''.join(str(num))))
    #cv2.imwrite(path_res, res)
    cv2.imwrite(path_mask, mask)
    #return mask

def make_colors():
    from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
    colors = []
    for cate in COCO_CATEGORIES:
        colors.append(cate["color"])
    return colors

def mask_to_boundary(mask, dilation_ratio=0.0008):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_entity_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="/home/lzr/projects/projects/Text-guided-image-editing/config/entity_swin_lw7_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        default= glob.glob('/home/lzr/projects/projects/Text-guided-image-editing/image/*.jpg'),
        type=list,
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='/home/lzr/projects/projects/Text-guided-image-editing/mask_output/',
        type=str,
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.4,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def get_seg_mask(image_path, mask_path):
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    colors = make_colors()

    img_list = []
    fileExtensions = [ "jpg", "jpeg", "png", "bmp", "gif" ]
    for extension in fileExtensions:
        img_list.extend(glob.glob(image_path + '/*' + extension))

    if img_list:
        if len(img_list) == 1:
            img_list = glob.glob(os.path.expanduser(img_list[0]))
            assert img_list, "The input path(s) was not found"
        for path in tqdm.tqdm(img_list, disable=not args.output):
            # use PIL, to be consistent with evaluation
            
            filename = path.split('/')[-1]
            (name, suffix) = filename.split('.')
            #print(name + suffix)
            img = read_image(path, format="BGR")
            start_time = time.time()
            data = demo.run_on_image_wo_vis(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(data[0])),
                    time.time() - start_time,
                )
            )

            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(path))
            else:
                assert len(img_list) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            ## save inference result, [0] original score by detection head, [1] mask rescoring score, [2] mask_id
            ori_scores = data[0]
            scores = data[1]
            mask_id = data[2]
            #np.savez(out_filename.split(".")[0]+".npz", ori_scores=ori_scores, scores=scores, mask_id=mask_id)

            ## save visualization
            img_for_paste = copy.deepcopy(img)
            color_mask     = copy.deepcopy(img)
            masks_edge     = np.zeros(img.shape[:2], dtype=np.uint8)
            
            alpha  = 0.4
            count  = 0
            #print(scores)
            for index, score in enumerate(scores):
                if score <= args.confidence_threshold:
                    break
                color_mask[mask_id==count] = colors[count]
                #print(colors[count])
                #cv2.imwrite('color_mask'+str(count)+".jpg", color_mask)
                generate_mask(name, color_mask, colors[count], count, mask_path)
                boundary = mask_to_boundary((mask_id==count).astype(np.uint8))
                masks_edge[boundary>0] = 0
                count += 1
                
