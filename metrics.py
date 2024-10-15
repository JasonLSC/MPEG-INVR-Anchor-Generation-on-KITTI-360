import glob
import argparse
import imageio
import numpy as np
import torch 
import scipy.signal
from tqdm import trange
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))

def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim

__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval()

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous()
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous()
    return __LPIPS__[net_name](gt, im, normalize=True).item()

parser = argparse.ArgumentParser()

parser.add_argument('--start_index',type = int,help=' Start index of Data')
parser.add_argument('--num_images',type = int,help='Num images')
config = parser.parse_args()

# glob to get all ref. png files
ref_png_filenames = []
all_ref_png_filenames = sorted(glob.glob(f"./nerfacto_{config.start_index}_{config.num_images}/*.png"))
for idx, filename in enumerate(all_ref_png_filenames):
    if idx % 4 == 2 or idx % 4 == 3:
        ref_png_filenames.append(filename)
torch_ref_imgs = torch.stack([torch.from_numpy(imageio.imread(name))/255. for name in ref_png_filenames]) #(N_img, H, W, 3)


metrics_dict = {}
for QP in ["RP0", "QP1", "QP2", "QP3", "QP4"]:
# glob to get all YUV files
    render_png_filenames = sorted(glob.glob(f"./nerfacto_{config.start_index}_{config.num_images}/tmiv_enc/{QP}/render/*.png"))
    torch_render_imgs = torch.stack([torch.from_numpy(imageio.imread(name))/255. for name in render_png_filenames]) #(N_img, H, W, 3)

    PSNRs = []
    SSIMs = []
    LPIPS_Alexs = []
    LPIPS_VGGs = []
    
    for i in trange(torch_ref_imgs.shape[0]):
        mse = torch.mean((torch_render_imgs[i] - torch_ref_imgs[i]) ** 2)
        psnr = mse2psnr(mse)

        ssim = rgb_ssim(torch_render_imgs[i], torch_ref_imgs[i], 1)
        l_a = rgb_lpips(torch_render_imgs[i].numpy(), torch_ref_imgs[i].numpy(), 'alex', device)
        l_v = rgb_lpips(torch_render_imgs[i].numpy(), torch_ref_imgs[i].numpy(), 'vgg', device)

        PSNRs.append(psnr)
        SSIMs.append(ssim)
        LPIPS_Alexs.append(l_a)
        LPIPS_VGGs.append(l_v)
    
    info_dict = {
        "PSNR": torch.stack(PSNRs).mean().item(),
        "SSIM": sum(SSIMs)/len(SSIMs),
        "LPIPS_A": sum(LPIPS_Alexs)/len(LPIPS_Alexs),
        "LPIPS_V": sum(LPIPS_VGGs)/len(LPIPS_VGGs)
    }
    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    bitstream_filenames = sorted(glob.glob(f"./nerfacto_{config.start_index}_{config.num_images}/tmiv_enc/{QP}/*.bit"))
    bitstream_size = sum([os.path.getsize(file_name) for file_name in bitstream_filenames])
    bitrate = bitstream_size/1e6*240 # Mbps

    info_dict.update(
        {"file_size": bitstream_size,
         "bitrate": bitrate}
    )

    print(QP, f"{bitstream_size/1e6} MB", f"{bitstream_size/1e6*240} Mbps")
    metrics_dict.update({
        QP: info_dict
    })

os.makedirs(f"./nerfacto_{config.start_index}_{config.num_images}/rd", exist_ok=True)

with open(f"./nerfacto_{config.start_index}_{config.num_images}/rd/rd.json", "w") as file:
    json.dump(metrics_dict, file)


# bitrate calculation

