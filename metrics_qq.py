#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
# from utils.loss_utils import ssim
# from lpipsPyTorch import lpips
# from utils.image_utils import psnr

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import json
from tqdm import tqdm
from argparse import ArgumentParser
import cv2
from torchvision import transforms

device = torch.device("cuda:0")
torch.cuda.set_device(device)

loss_fn = lpips.LPIPS(net='alex').to(device)

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        # render = Image.open(renders_dir / fname)
        # gt = Image.open(gt_dir / fname)
        # renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        # gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        renders.append(renders_dir / fname)
        gts.append(gt_dir / fname)
        image_names.append(fname)
    return renders, gts, image_names


def calculate_metrics(render_path, gt_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    计算单个图像对的SSIM、PSNR和LPIPS指标
    
    参数:
    render_path: 渲染图像路径
    gt_path: 真实图像路径
    device: 计算设备，默认为GPU
    
    返回:
    ssim_val: SSIM值
    psnr_val: PSNR值
    lpips_val: LPIPS值
    """
    # 读取图像
    render_img = cv2.imread(render_path)
    gt_img = cv2.imread(gt_path)
    
    # 确保图像尺寸一致
    if render_img.shape != gt_img.shape:
        gt_img = cv2.resize(gt_img, (render_img.shape[1], render_img.shape[0]))

    # 计算SSIM (使用skimage)
    ssim_val = ssim(render_img, gt_img, multichannel=True, data_range=255,win_size=3)
    
    # 计算PSNR (使用skimage)
    psnr_val = psnr(gt_img, render_img, data_range=255)
    
    # print(ssim_val, psnr_val)
    # 计算LPIPS
    # loss_fn = lpips.LPIPS(net='alex').to(device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 加载图像并预处理
    render_pil = Image.fromarray(render_img)
    gt_pil = Image.fromarray(gt_img)
    
    render_tensor = transform(render_pil).unsqueeze(0).to(device)
    gt_tensor = transform(gt_pil).unsqueeze(0).to(device)
    
    # 计算LPIPS距离
    with torch.no_grad():
        lpips_val = loss_fn(render_tensor, gt_tensor).item()
    
    
    return ssim_val, psnr_val, lpips_val

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress", dynamic_ncols=True, ascii=True):
                    # print()
                    # ssims.append(ssim(renders[idx], gts[idx]))
                    # print(ssim(renders[idx], gts[idx]))
                    # psnrs.append(psnr(renders[idx], gts[idx]))
                    # print(psnr(renders[idx], gts[idx]))
                    # lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
                    # print(lpips_fn(renders[idx], gts[idx]).detach())
                    ssim_val, psnr_val, lpips_val = calculate_metrics(renders[idx], gts[idx])
                    ssims.append(ssim_val)
                    psnrs.append(psnr_val)
                    lpipss.append(lpips_val)

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results_qq.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
