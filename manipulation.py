from glob import glob
import imp
from sys import float_repr_style
from typing import Tuple
from imageio import imread
from IPython.display import display
from PIL import Image
import numpy as np
import torch as th
import torch.nn.functional as F
from skimage.color import rgb2gray, gray2rgb
from glide_text2im.download import load_checkpoint
import torchvision
from detectron2.utils.logger import setup_logger
import os
import argparse
from GenerateEntity import get_Entity_with_mask
from GenerateMask import get_seg_mask
from EntitywithCLIP import return_mask
from inpaint import Use_Glide
import re

def get_parser():
    parser = argparse.ArgumentParser(description="Console")
    parser.add_argument(
        "--image_file",
        default="/home/lzr/projects/projects/Text-guided-image-editing/image/",
        metavar="FILE",
        help="path to original image file, hope it can matain different types of imgs",
    )

    parser.add_argument(
        "--mask_file",
        default="/home/lzr/projects/projects/Text-guided-image-editing/mask_output/",
        metavar="FILE",
        help="path to generated masks file : jpgs",
    )

    parser.add_argument(
        "--entity_file",
        default="/home/lzr/projects/projects/Text-guided-image-editing/Entity_Results/",
        metavar="FILE",
        help="path to generated entity_with_masks file, jpgs",
    )

    parser.add_argument(
        "--final_mask_path",
        default="/home/lzr/projects/projects/Text-guided-image-editing/Glide_results/",
        metavar="FILE",
        help="path to save final results, jpgs",
    )

    parser.add_argument(
        "--text",
        default= 'I want to remove the {beer} of the picture, The scenarios for the new images are {}',
        type=str,
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )

    parser.add_argument(
        "--CLIP_threshold",
        default= 0.2,
        type=float,
        help="CLIP_threshold",
    )
    return parser

def remove_all(mask_file, entity_file, final_mask_path):
    
    filelist = os.listdir(mask_file)
    for file in filelist:
        if '.jpg' in file:
            del_file = mask_file + file 
            os.remove(del_file)
            print("removed: ",del_file)

    filelist = os.listdir(entity_file)
    for file in filelist:
        if '.jpg' in file:
            del_file = entity_file + file 
            os.remove(del_file)
            print("removed: ",del_file)

    filelist = os.listdir(final_mask_path)
    for file in filelist:
        if '.jpg' in file:
            del_file = final_mask_path + file 
            os.remove(del_file)
            print("removed: ",del_file)

def get_contentoftext(text: str):
    res = re.findall(r'\{(.*?)\}',text)
    return res

if __name__ == "__main__":

    # load parser and print the informations
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    descriptions = get_contentoftext(args.text)

    descriptions_clip = 'a picture of ' + descriptions[0]

    descriptions_diffusion = descriptions[1]

    # get_seg_mask(args.image_file, args.mask_file)
    
    name_lsits = get_Entity_with_mask(args.image_file, args.mask_file, args.entity_file)

    mask_256 = return_mask(descriptions_clip, args.entity_file, args.CLIP_threshold, name_lsits[0], args.mask_file)

    Use_Glide(descriptions_diffusion, args.image_file, args.mask_file)

    #remove_all(args.mask_file, args.entity_file, args.final_mask_path)
    
    