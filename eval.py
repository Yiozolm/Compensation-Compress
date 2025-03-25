import math
import os
import sys
import time
from collections import defaultdict
from typing import List, Dict, Any
from PIL import Image
# To write result in xlsx files
from openpyxl import Workbook
import numpy as np
import re

import torch
import torch.nn.functional as F

# Metrics
from DISTS_pytorch import DISTS
from pytorch_fid import fid_score
from pytorch_msssim import ms_ssim
import lpips

from models import Compensation

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    I = np.array(img).astype(np.float32)
    output = torch.as_tensor(I).permute(2, 0, 1)
    return output


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics(
        org: torch.Tensor, rec: torch.Tensor, max_val: int = 255
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    org = org.round()
    rec = rec.round()
    metrics["psnr-rgb"] = psnr(org, rec).item()
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    metrics["log_ssim"] = -10 * np.log10(1 - metrics["ms-ssim"])
    return metrics


@torch.no_grad()
def inference(model, x, recon_dir, name=None):
    x = x.unsqueeze(0)

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    x_padded = x_padded / 255.
    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    out_dec["x_hat"] = out_dec["x_hat"] * 255.
    if name:
        out = out_dec["x_hat"].clone().squeeze(0)
        out = out.permute(1, 2, 0)
        out = out.cpu().numpy()

        # print(out.shape)
        I_ll = out.astype(np.uint8)
        im_nll = Image.fromarray(I_ll)
        im_nll.save(recon_dir + '/' + name[-11:])

    metrics = compute_metrics(x, out_dec["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    return {
        "psnr": metrics["psnr-rgb"],
        "ms-ssim": metrics["ms-ssim"],
        "log_ssim": metrics["log_ssim"],
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


@torch.no_grad()
def inference_entropy_estimation(model, x):
    x = x.unsqueeze(0)
    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    x_padded = x_padded / 255.
    start = time.time()
    out_net = model.forward(x_padded)
    elapsed_time = time.time() - start

    out_net["x_hat"] = F.pad(
        out_net["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    out_net["x_hat"] = out_net["x_hat"] * 255.
    metrics = compute_metrics(x, out_net["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    return {
        "psnr": metrics["psnr-rgb"],
        "ms-ssim": metrics["ms-ssim"],
        "log_ssim": metrics["log_ssim"],
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def eval_model(model, filepaths, recon_dir, entropy_estimation=False, half=False):
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    for f in filepaths:
        # print(f)
        x = read_image(f).to(device)
        if not entropy_estimation:
            if half:
                model = model.to(torch.bfloat16)
                x = x.to(torch.bfloat16)
            rv = inference(model, x, recon_dir, f)
        else:
            rv = inference_entropy_estimation(model, x)
        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics


def eval_dist(origin_path, gene_path):
    """
    Specific details available at "https://github.com/dingkeyan93/DISTS"
    :param origin_path: the original datasets root path
    :param gene_path: the reconstructed datasets root path
    :return: average DISTS value
    """
    device = torch.device('cuda')
    D = DISTS(load_weights=True).to(device)
    files = os.listdir(origin_path)
    distSum = 0
    for f in files:
        x = read_image(os.path.join(origin_path, f)).to(device)
        x = x / 255.
        y = read_image(os.path.join(gene_path, f)).to(device)
        y = y / 255.
        x.unsqueeze_(0)
        y.unsqueeze_(0)
        dists_value = D(x, y)
        distSum += dists_value.item()
    del D

    return distSum / len(files)


def eval_fid(origin_path, gene_path):
    """
    Specific details available at "https://github.com/mseitzer/pytorch-fid"
    :param origin_path: the original datasets root path
    :param gene_path: the reconstructed datasets root path
    :return: average FID value
    """
    fid_value = fid_score.calculate_fid_given_paths([origin_path, gene_path],
                                                    batch_size=24, device='cuda:0', dims=2048)
    return fid_value


def eval_lpips(origin_path, gene_path):
    """
    Specific details available at "https://github.com/richzhang/PerceptualSimilarity"
    :param origin_path: the original datasets root path
    :param gene_path: the reconstructed datasets root path
    :return: average LPIPS value
    """
    loss_fn = lpips.LPIPS(net='alex', version='0.1')
    loss_fn.cuda()
    files = os.listdir(origin_path)
    i = 0
    total_lpips_distance = 0

    for file in files:

        try:
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(origin_path, file))).cuda()
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(gene_path, file))).cuda()

            if (os.path.exists(os.path.join(origin_path, file)), os.path.exists(os.path.join(gene_path, file))):
                i = i + 1

            # Compute distance
            current_lpips_distance = loss_fn.forward(img0, img1)
            total_lpips_distance = total_lpips_distance + current_lpips_distance.item()

            # print('%s: %.3f'%(file, current_lpips_distance))

        except Exception as e:
            print(e)

    del loss_fn
    return total_lpips_distance / i


def eval_ckp(model, ckp_name, filepaths, photo_dir, recon_dir):
    metrics1 = eval_model(model, filepaths, recon_dir, entropy_estimation=False)
    metrics2 = eval_model(model, filepaths, recon_dir, entropy_estimation=True)
    FIDvalue = eval_fid(photo_dir, recon_dir)
    LPIPSvalue = eval_lpips(photo_dir, recon_dir)
    DISTSvalue = eval_dist(photo_dir, recon_dir)
    row = [int(re.findall(r'\d+', ckpname)[0]) / 10000, metrics1['bpp'], metrics2['bpp'], abs(metrics1['bpp'] - metrics2['bpp']),
           metrics1['psnr'], metrics2['psnr'], abs(metrics1['psnr'] - metrics2['psnr']), metrics1['ms-ssim'],
           metrics2['ms-ssim'], abs(metrics1['ms-ssim'] - metrics2['ms-ssim']), FIDvalue, LPIPSvalue, DISTSvalue]
    return row


if __name__ == "__main__":
    # load image files
    photo_dir = "./Kodak24"
    filepaths = collect_images(photo_dir)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(1)

    # Create a new workbook and select the active worksheet
    workbook = Workbook()
    sheet = workbook.active
    rows = []

    # load checkpoint files
    ckp_dir = "./ckp_ll/our/"
    ckps = os.listdir(ckp_dir)
    device = torch.device('cuda')
    YourModelName = ""
    assert YourModelName is not None, "Please specify a model name."

    for ckpname in ckps:
        lamda = int(re.findall(r'\d+', ckpname)[0])
        print(f"-----Processing ckp-{lamda}-----")
        if lamda < 150:
            model = Compensation(N=128, M=192).cuda()   # add your model here
        else:
            model = Compensation(N=192, M=320).cuda()
        ckp = torch.load(os.path.join(ckp_dir, ckpname), map_location=device)
        model.load_state_dict(ckp)
        model.update(force=True)

        recon_dir = f'./Kodaktest/{YourModelName}/{YourModelName}-' + str(lamda)
        if not os.path.exists(recon_dir):
            os.makedirs(recon_dir)
        row = eval_ckp(model, ckpname, filepaths, photo_dir, recon_dir)
        rows.append(row)
        print('-----Processing End-----')

    sheet.append(
        ['λ', 'bpp', 'bpp-est', 'Δbpp', 'psnr', 'psnr-est', 'Δpsnr', 'ms-ssim', 'ms-ssim-est', 'Δms-ssim', 'FID',
         'LPIPS', 'DISTS'])
    rows = sorted(rows, key=lambda x: x[0])
    for row in rows:
        sheet.append(row)
    workbook.save(f'./Kodaktest/{YourModelName}/{YourModelName}.xlsx')
