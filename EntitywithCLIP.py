import argparse
from email import parser
from glob import glob
import torch
import clip
from PIL import Image
import glob
import imageio
import torchvision
import numpy as np
from imageio import imread
from skimage.color import rgb2gray, gray2rgb
from skimage import io
import cv2
import re

parser = argparse.ArgumentParser(description='Entity with CLIP')
# function UseClip
parser.add_argument('--original_image_path', type=str, default='/home/lzr/projects/Text-guided-image-editing/image/2012_001298.jpg', help='original_image_path, single')
parser.add_argument('--entity_path', type=str, default='/home/lzr/projects/Text-guided-image-editing/Entity_Results/', help='many imgs, use glob to handle this')
parser.add_argument('--text', type=str, default='a picture of beer', help='many imgs, use glob to handle this')
parser.add_argument('--final_mask_path', type=str, default='/home/lzr/projects/Text-guided-image-editing/output/', help='many imgs, use glob to handle this')
parser.add_argument('--mask_path', type=str, default='/home/lzr/projects/Text-guided-image-editing/mask_output/', help='many imgs, use glob to handle this')

# function find_mask
parser.add_argument('--threshold', type=float, default=0.3)
opt = parser.parse_args()



def UseClip(text, entity_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_inputs=[]
    text = clip.tokenize([text]).to(device)

    image_paths = sorted(glob.glob(entity_path + '*.jpg'))
    #print(image_paths)
    for image_path in image_paths:
        img = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_inputs.append(img)

    img_inpuit = torch.concat(image_inputs, dim=0)
    with torch.no_grad():
        #image_features = model.encode_image(torch.stack(image_inputs))
        logits_per_image, logits_per_text = model(img_inpuit, text)
        probs = logits_per_text.softmax(dim=-1).cpu().numpy()

    return probs[0], image_paths


def find_mask(probs, name, image_paths, mask_path, CLIP_threshold):
    # return a list of choosen entities with masks
    imgs = []
    for (prob, img) in zip(probs, sorted(image_paths)):
        if prob > CLIP_threshold:
            imgs.append(img)
    numbers = []

    for img in imgs:
        # pattern = re.compile(r'(?<=entity_)\d+')
        # number = pattern.findall(img)    # 输出结果为列表
        file_name = img.split('/')[-1]
        x = file_name.replace('entity','mask')
        str = mask_path + x
        numbers.append(str)
    return numbers

def read_image(path: str):
    pil_img = Image.open(path).convert('RGB')
    img = np.array(pil_img)
    return torch.from_numpy(img)[None].permute(0, 3, 1, 2).float()

def to_tensor(img):
    img = Image.fromarray(img)
    img_t = torchvision.transforms.functional.to_tensor(img).float()
    return img_t

def load_mask(path: str):
    
    mask = imageio.v2.imread(path)
    #mask = rgb2gray(mask)
    #mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
    return 255 - mask

def resize(img, height, width, centerCrop=True):
    imgh, imgw = img.shape[0:2]

    if centerCrop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]

    img = np.array(Image.fromarray(img).resize((height,width)))

    return img

def return_mask(text, entity_paths, CLIP_threshold, name, mask_file):
    
    probs, image_paths = UseClip(text, entity_paths)
    mask_paths = find_mask(probs, name, image_paths, mask_file, CLIP_threshold)


    pixel_max_cnt = 255
    # img = imageio.v2.imread(original_image_path)
    # img = resize(img, H, W)
    # img = to_tensor(img)
    #print(img.shape)
    img = load_mask(mask_paths[0])
    img_H, img_W  = img.shape[0], img.shape[1]
    source_mask = torch.ones_like(torch.empty(1, img_H, img_W))
    #print(source_mask.shape)
    for mask_path in mask_paths:
        mask = load_mask(mask_path)
        #mask = resize(mask, H, W)
        mask = to_tensor(mask)
        
        source_mask = source_mask * mask

    return_data = source_mask

    return_data = return_data * 255
    return_data = 255 - return_data
    img_copy = return_data.clone().data.permute(1,2,0).cpu().numpy()
    #img_copy = np.clip(img_copy, 0, pixel_max_cnt)
    img_copy = img_copy.astype(np.uint8)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    cv2.imwrite(mask_file + 'combine_mask.jpg', img_copy)

    return source_mask



if __name__ == "__main__":

    probs, image_paths = UseClip(opt.text, opt.entity_path)
    mask_paths = find_mask(probs, '100sh', image_paths, opt.mask_path, 0.3)
    # print(probs)
    # print('______________________')
    # print(image_paths)
    # print('______________________')
    # print(mask_paths)


    # return_mask('beer', opt.entity_path, 0.2, '100sh', opt.mask_path)


    # pixel_max_cnt = 255
    # img = imageio.v2.imread(opt.original_image_path)
    # img = to_tensor(img)
    # print(img.shape)
    # source_mask = torch.ones_like(img[:1,:])
    # print(source_mask.shape)
    # for mask_path in mask_paths:
    #     mask = load_mask(mask_path)
    #     mask = to_tensor(mask)
        
    #     source_mask = source_mask * mask
    # #source_mask = np.clip(source_mask, 0, pixel_max_cnt)
    # img = img * source_mask
    
    # img = img * 255.0
    
    # img_copy = img.clone().data.permute(1,2,0).cpu().numpy()
    # #img_copy = np.clip(img_copy, 0, pixel_max_cnt)
    # img_copy = img_copy.astype(np.uint8)
    # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(opt.final_mask_path + 'result.jpg', img_copy)

    # x = img.round().clamp(0,255).to(torch.uint8).cpu()
    # y = x.permute(2, 0, 3, 1).reshape([(img).shape[2], -1, 3])
    # Image.fromarray(y.numpy()).save(opt.final_mask_path + 'result.jpg')




