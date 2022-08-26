from glob import glob
import imp
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
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)
import argparse
from GenerateEntity import get_Entity_with_mask
from GenerateMask import get_seg_mask
from EntitywithCLIP import return_mask
from torchvision.transforms import Resize

parser = argparse.ArgumentParser(description='impaint using Glide')
# function UseClip
parser.add_argument('--original_image_path', type=str, default='/home/lzr/projects/Text-guided-image-editing/image/2012_001298.jpg', help='original_image_path, single')
parser.add_argument('--entity_paths', type=str, default='/home/lzr/projects/Text-guided-image-editing/Entity_Results/', help='many imgs, use glob to handle this')
parser.add_argument('--text', type=str, default='A man is riding his bicycle on a country road', help='many imgs, use glob to handle this')
parser.add_argument('--final_mask_path', type=str, default='/home/lzr/projects/Text-guided-image-editing/output/', help='many imgs, use glob to handle this')
parser.add_argument('--mask_path', type=str, default='/home/lzr/projects/Text-guided-image-editing/mask_output/', help='many imgs, use glob to handle this')
opt = parser.parse_args()


def Use_Glide(text, image_path, mask_path):
    # This notebook supports both CPU and GPU.
    # On CPU, generating one sample may take on the order of 20 minutes.
    # On a GPU, it should be under a minute.

    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')



    # Create base model.
    options = model_and_diffusion_defaults()
    options['inpaint'] = True
    options['use_fp16'] = has_cuda
    options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    model.load_state_dict(load_checkpoint('base-inpaint', device))
    print('total base parameters', sum(x.numel() for x in model.parameters()))



    # Create upsampler model.
    options_up = model_and_diffusion_defaults_upsampler()
    options_up['inpaint'] = True
    options_up['use_fp16'] = has_cuda
    options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if has_cuda:
        model_up.convert_to_fp16()
    model_up.to(device)
    model_up.load_state_dict(load_checkpoint('upsample-inpaint', device))
    print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

    def show_images(batch: th.Tensor):
        """ Display a batch of images inline. """
        scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
        reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
        display(Image.fromarray(reshaped.numpy()))

    def read_image(path: str, size: int = 256) -> Tuple[th.Tensor, th.Tensor]:
        pil_img = Image.open(path).convert('RGB')
        pil_img = pil_img.resize((size, size), resample=Image.BICUBIC)
        img = np.array(pil_img)
        return th.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1
    
    def read_mask(path: str, size: int = 256) -> Tuple[th.Tensor, th.Tensor]:
        pil_img = Image.open(path).convert('RGB')
        pil_img = pil_img.resize((size, size), resample=Image.BICUBIC)
        img = np.array(pil_img)
        #img = (img > 0).astype(np.uint8) * 255
        img = 255 - img
        return th.from_numpy(img)[None].float() 
    
        
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

    def load_mask(path):
        mask = imread(path)   # mask must be 255 for hole in this InpaintingModel
        #mask = resize(mask, imgh, imgw, centerCrop=False)
        #mask = rgb2gray(mask)
        #mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
        mask = 255 - mask
        return mask

    def to_tensor(img):
        img = Image.fromarray(img)
        img_t = torchvision.transforms.functional.to_tensor(img).float()
        return img_t


    # Sampling parameters
    prompt = text
    batch_size = 1
    guidance_scale = 5.0

    # Tune this parameter to control the sharpness of 256x256 images.
    # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
    upsample_temp = 0.997



    # Segmentation of entities using existing models,
    # The threshold value has been adjusted to 0.4
    # get_seg_mask(opt.mask_path)

    # # Combine the original image with the split mask and output the result at the specified file
    # maskpath_splitting = sorted(glob(opt.mask_path + '/*.jpg'))
    # get_Entity_with_mask(opt.original_image_path, maskpath_splitting, opt.entity_path)


    # source_mask_256 = created_mask
    # source_mask_64 = F.interpolate(source_mask_256.unsqueeze(0), size=(64, 64), mode='nearest').squeeze(0)
    mk_path = mask_path + 'combine_mask.jpg'
    img_path = glob(image_path + '/*.jpg')
    
    mask = load_mask(mk_path)
    img_H, img_W  = mask.shape[0], mask.shape[1]
    source_mask = th.ones_like(th.empty(1, img_H, img_W))
    #print(source_mask.shape)
     #mask = resize(mask, H, W)
    mask = to_tensor(mask)
        
    source_mask = source_mask * mask
    torch_resize_256 = Resize([256,256])
    torch_resize_64 = Resize([64,64])

    source_mask_256 = torch_resize_256(source_mask).unsqueeze(dim=0)[:,:1,:]
    source_mask_64 = torch_resize_64(source_mask).unsqueeze(dim=0)[:,:1,:]
    print(source_mask_64.shape)
    # source_mask_256 = read_mask(mk_path, size=256)
    # #source_mask_256 = source_mask_256[:, :1]

    # source_mask_64 = read_mask(mk_path, size=64)
    # #source_mask_64 = source_mask_64[:, :1]
    
    source_image_256 = read_image(img_path[0], size=256)
    source_image_64 = read_image(img_path[0], size=64)
    print(source_image_64.shape)

    # The mask should always be a boolean 64x64 mask, and then we
    # can upsample it for the second stage.

    # source_mask_64 = source_mask.reshape([64, 64])
    # source_mask_256 = source_mask.reshape([256, 256])

    # Visualize the image we are inpainting
    #show_images(source_image_256 * source_mask_256)

    x = ((source_image_256 * source_mask_256 + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    y = x.permute(2, 0, 3, 1).reshape([(source_image_256 * source_mask_256).shape[2], -1, 3])
    Image.fromarray(y.numpy()).save('./Glide_results/result_mask.jpg')





    ##############################
    # Sample from the base model #
    ##############################

    # Create the text tokens to feed to the model.
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(
        tokens, options['text_ctx']
    )

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options['text_ctx']
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=th.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),

        # Masked inpainting image
        inpaint_image=(source_image_64 * source_mask_64).repeat(full_batch_size, 1, 1, 1).to(device),
        inpaint_mask=source_mask_64.repeat(full_batch_size, 1, 1, 1).to(device),
    )

    # Create an classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)

    def denoised_fn(x_start):
        # Force the model to have the exact right x_start predictions
        # for the part of the image which is known.
        return (
            x_start * (1 - model_kwargs['inpaint_mask'])
            + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask']
        )

    # Sample from the base model.
    model.del_cache()
    samples = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
        denoised_fn=denoised_fn,
    )[:batch_size]
    model.del_cache()

    # Show the output
    show_images(samples)
    ##############################
    # Upsample the 64x64 samples #
    ##############################

    tokens = model_up.tokenizer.encode(prompt)
    tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
        tokens, options_up['text_ctx']
    )

    # Create the model conditioning dict.
    model_kwargs = dict(
        # Low-res image to upsample.
        low_res=((samples+1)*127.5).round()/127.5 - 1,

        # Text tokens
        tokens=th.tensor(
            [tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),

        # Masked inpainting image.
        inpaint_image=(source_image_256 * source_mask_256).repeat(batch_size, 1, 1, 1).to(device),
        inpaint_mask=source_mask_256.repeat(batch_size, 1, 1, 1).to(device),
    )

    def denoised_fn(x_start):
        # Force the model to have the exact right x_start predictions
        # for the part of the image which is known.
        return (
            x_start * (1 - model_kwargs['inpaint_mask'])
            + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask']
        )

    # Sample from the base model.
    model_up.del_cache()
    up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
    up_samples = diffusion_up.p_sample_loop(
        model_up,
        up_shape,
        noise=th.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
        denoised_fn=denoised_fn,
    )[:batch_size]
    model_up.del_cache()

    # Show the output
    show_images(up_samples)
    scaled = ((up_samples + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([up_samples.shape[2], -1, 3])
    Image.fromarray(reshaped.numpy()).save('./Glide_results/result.jpg')