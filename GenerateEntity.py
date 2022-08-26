from cgitb import reset
from PIL import Image
import torch
import numpy as np
import imageio
from IPython.display import display
import torchvision.transforms.functional
import glob


def read_image(path: str):
    pil_img = Image.open(path).convert('RGB')
    img = np.array(pil_img)
    return torch.from_numpy(img)[None].permute(0, 3, 1, 2).float()

def load_mask(path: str):
    mask = imageio.v2.imread(path)   # mask must be 255 for hole in this InpaintingModel
    #mask = rgb2gray(mask)
    #mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
    #mask = 255 - mask
    return to_tensor(mask)

def to_tensor(img):
    img = Image.fromarray(img)
    img_t = torchvision.transforms.functional.to_tensor(img).float()
    return img_t

def get_Entity_with_mask(image_path:str, mask_paths:str, out_path:str):
    imgpath_list = []
    fileExtensions = [ "jpg", "jpeg", "png", "bmp", "gif" ]

    for extension in fileExtensions:
        imgpath_list.extend(glob.glob(image_path + '/*' +extension))

    mask_paths = sorted(glob.glob(mask_paths + '/*.jpg'))

    name_lsit = []
    for imgpath in imgpath_list:
        filename = imgpath.split('/')[-1]
        (name, suffix) = filename.split('.')
        
        img = read_image(imgpath)
        count = 0
        for mask_path in mask_paths:
            if name in mask_path:
                mask = load_mask(mask_path)
                result = mask * img
                x = result.round().clamp(0,255).to(torch.uint8).cpu()
                y = x.permute(2, 0, 3, 1).reshape([(result).shape[2], -1, 3])
                Image.fromarray(y.numpy()).save(out_path + '/' +name + '_entity_{}.jpg'.format(''.join(str(count))))
                count += 1
                name_lsit.append(name)
    return name_lsit


if __name__ == "__main__":
    path = '/home/lzr/projects/Text-guided-image-editing/image/2012_001298.jpg'
    mask_paths = sorted(glob.glob('/home/lzr/projects/Text-guided-image-editing/mask_output/*.jpg'))
    