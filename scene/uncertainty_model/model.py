import hashlib
import struct
from typing import Optional, cast, Any, TypeVar
import logging
import io
import itertools
import random
from tqdm import tqdm
from functools import reduce
from operator import mul
import urllib.request
import itertools
from typing import Optional, cast, Any, TypeVar
import random
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from omegaconf import OmegaConf
import os
import math
import scene.uncertainty_model.dinov2 as dinov2
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from matplotlib.gridspec import GridSpec



from scene.uncertainty_model.my_types import (
    Method,
    MethodInfo,
    RenderOutput,
    ModelInfo,
    camera_model_to_int,
    Dataset,
    Cameras,
    GenericCameras,
    OptimizeEmbeddingOutput,
)


class Config:
    uncertainty_mode: str = "disabled"
    uncertainty_backbone: str = "dinov2_vits14"
    uncertainty_regularizer_weight: float = 0.5
    uncertainty_clip_min: float = 0.1
    uncertainty_mask_clip_max: Optional[float] = None
    uncertainty_dssim_clip_max: float = 1.0  # 0.05 -> 0.005
    uncertainty_lr: float = 0.001
    uncertainty_dropout: float = 0.5
    uncertainty_dino_max_size: Optional[int] = None
    uncertainty_scale_grad: bool = False
    uncertainty_center_mult: bool = False
    uncertainty_after_opacity_reset: int = 1000
    uncertainty_protected_iters: int = 500
    uncertainty_preserve_sky: bool = False

    uncertainty_warmup_iters: int = 0
    uncertainty_warmup_start: int = 1000


T = TypeVar("T")


def convert_image_dtype(image: np.ndarray, dtype) -> np.ndarray:
    if image.dtype == dtype:
        return image
    if image.dtype != np.uint8 and dtype != np.uint8:
        return image.astype(dtype)
    if image.dtype == np.uint8 and dtype != np.uint8:
        return image.astype(dtype) / 255.0
    if image.dtype != np.uint8 and dtype == np.uint8:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)
    raise ValueError(f"cannot convert image from {image.dtype} to {dtype}")


def get_torch_checkpoint_sha(checkpoint_data):
    sha = hashlib.sha256()

    def update(d):
        if type(d).__name__ == "Tensor" or type(d).__name__ == "Parameter":
            sha.update(d.cpu().numpy().tobytes())
        elif isinstance(d, dict):
            items = sorted(d.items(), key=lambda x: x[0])
            for k, v in items:
                update(k)
                update(v)
        elif isinstance(d, (list, tuple)):
            for v in d:
                update(v)
        elif isinstance(d, (int, float)):
            sha.update(struct.pack("f", d))
        elif isinstance(d, str):
            sha.update(d.encode("utf8"))
        elif d is None:
            sha.update("(None)".encode("utf8"))
        else:
            raise ValueError(f"Unsupported type {type(d)}")

    update(checkpoint_data)
    return sha.hexdigest()


def assert_not_none(value: Optional[T]) -> T:
    assert value is not None
    return value


def camera_project(cameras: GenericCameras[Tensor], xyz: Tensor) -> Tensor:
    eps = torch.finfo(xyz.dtype).eps  # type: ignore
    assert xyz.shape[-1] == 3

    # World -> Camera
    origins = cameras.poses[..., :3, 3]
    rotation = cameras.poses[..., :3, :3]
    # Rotation and translation
    uvw = xyz - origins
    uvw = (rotation * uvw[..., :, None]).sum(-2)

    # Camera -> Camera distorted
    uv = torch.where(
        uvw[..., 2:] > eps, uvw[..., :2] / uvw[..., 2:], torch.zeros_like(uvw[..., :2])
    )

    # We assume pinhole camera model in 3DGS anyway
    # uv = _distort(cameras.camera_models, cameras.distortion_parameters, uv, xnp=xnp)

    x, y = torch.moveaxis(uv, -1, 0)

    # Transform to image coordinates
    # Camera distorted -> Image
    fx, fy, cx, cy = torch.moveaxis(cameras.intrinsics, -1, 0)
    x = fx * x + cx
    y = fy * y + cy
    return torch.stack((x, y), -1)


def safe_state():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def scale_grads(values, scale):
    grad_values = values * scale
    rest_values = values.detach() * (1 - scale)
    return grad_values + rest_values


# SSIM
def ssim(img1, img2, window_size=11, size_average=True):
    sigma = 1.5
    channel = img1.size(-3)
    # Create window
    gauss = torch.Tensor(
        [
            math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(-3)


def ssim_down(x, y, max_size=None):
    osize = x.shape[2:]
    if max_size is not None:
        scale_factor = max(max_size / x.shape[-2], max_size / x.shape[-1])
        x = F.interpolate(x, scale_factor=scale_factor, mode="area")
        y = F.interpolate(y, scale_factor=scale_factor, mode="area")
    out = ssim(x, y, size_average=False).unsqueeze(1)
    if max_size is not None:
        out = F.interpolate(out, size=osize, mode="bilinear", align_corners=False)
    return out.squeeze(1)


def _ssim_parts(img1, img2, window_size=11):
    sigma = 1.5
    channel = img1.size(-3)
    # Create window
    gauss = torch.Tensor(
        [
            math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )
    sigma1 = torch.sqrt(sigma1_sq.clamp_min(0))
    sigma2 = torch.sqrt(sigma2_sq.clamp_min(0))

    C1 = 0.01**2
    C2 = 0.03**2
    C3 = C2 / 2

    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    structure = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    return luminance, contrast, structure


def msssim(x, y, max_size=None, min_size=200):
    raw_orig_size = x.shape[-2:]
    if max_size is not None:
        scale_factor = min(1, max(max_size / x.shape[-2], max_size / x.shape[-1]))
        x = F.interpolate(x, scale_factor=scale_factor, mode="area")
        y = F.interpolate(y, scale_factor=scale_factor, mode="area")

    ssim_maps = list(_ssim_parts(x, y))
    orig_size = x.shape[-2:]
    while x.shape[-2] > min_size and x.shape[-1] > min_size:
        x = F.avg_pool2d(x, 2)
        y = F.avg_pool2d(y, 2)
        ssim_maps.extend(
            tuple(
                F.interpolate(x, size=orig_size, mode="bilinear")
                for x in _ssim_parts(x, y)[1:]
            )
        )
    out = torch.stack(ssim_maps, -1).prod(-1)
    if max_size is not None:
        out = F.interpolate(out, size=raw_orig_size, mode="bilinear")
    return out.mean(1)


def dino_downsample(x, max_size=None):
    if max_size is None:
        return x
    h, w = x.shape[2:]
    if max_size < h or max_size < w:
        scale_factor = min(max_size / x.shape[-2], max_size / x.shape[-1])
        nh = int(h * scale_factor)
        nw = int(w * scale_factor)
        nh = ((nh + 13) // 14) * 14
        nw = ((nw + 13) // 14) * 14
        x = F.interpolate(x, size=(nh, nw), mode="bilinear")
    return x


def assert_not_none(value: Optional[T]) -> T:
    assert value is not None
    return value

# jqq 可视化中间结果
def visualize_all_tensors3(
    gt,                     # 新增：原始图像
    prediction,             # 新增：预测图像
    uncertainty, 
    dino_cosine, 
    dino_part, 
    uncertainty_loss, 
    loss_mult1, 
    loss_mult2, 
    loss_mult3,
    loss_mult4,
    save_path=None, 
    show=True
):
    """
    可视化多个张量并拼接为单张图片（基于原始数据范围映射颜色），新增gt和prediction的可视化
    
    参数:
        gt: 原始图像张量
        prediction: 预测图像张量
        uncertainty: 不确定性张量
        dino_cosine: DINO余弦相似度张量
        dino_part: 处理后的DINO相似度张量
        uncertainty_loss: 不确定性损失张量
        loss_mult1: 损失乘数1
        loss_mult2: 损失乘数2
        loss_mult3: 损失乘数3
        save_path: 保存路径，若为None则不保存
        show: 是否显示图像
    """

    # 设置中文字体（支持中文显示）
    # plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    # # 解决负号显示问题
    # plt.rcParams["axes.unicode_minus"] = False
    # 调整全局字体大小（标题、标签等）
    plt.rcParams["font.size"] = 5

    # 确保所有张量在CPU上并转换为numpy数组
    tensors = {
        'gt': gt.cpu().numpy() if isinstance(gt, torch.Tensor) else gt,
        'prediction': prediction.cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction,
        'uncertainty': uncertainty.cpu().numpy() if isinstance(uncertainty, torch.Tensor) else uncertainty,
        'dino_cosine': dino_cosine.cpu().numpy() if isinstance(dino_cosine, torch.Tensor) else dino_cosine,
        'dino_part': dino_part.cpu().numpy() if isinstance(dino_part, torch.Tensor) else dino_part,
        'uncertainty_loss': uncertainty_loss.cpu().numpy() if isinstance(uncertainty_loss, torch.Tensor) else uncertainty_loss,
        'loss_mult1': loss_mult1.cpu().numpy() if isinstance(loss_mult1, torch.Tensor) else loss_mult1,
        'loss_mult2': loss_mult2.cpu().numpy() if isinstance(loss_mult2, torch.Tensor) else loss_mult2,
        'loss_mult3': loss_mult3.cpu().numpy() if isinstance(loss_mult3, torch.Tensor) else loss_mult3,
        'loss_mult4': loss_mult4.cpu().numpy() if isinstance(loss_mult4, torch.Tensor) else loss_mult4
    }
    
    # 确保所有张量是2D或3D（图像数据为3D时保持维度）
    for key in tensors:
        if key in ['gt', 'prediction']:
            if tensors[key].ndim == 4 and tensors[key].shape[0] == 1:
                tensors[key] = tensors[key].squeeze(0)
            if tensors[key].ndim == 3 and tensors[key].shape[0] == 3:
                tensors[key] = np.moveaxis(tensors[key], 0, -1)  # 转换为HWC格式
        else:
            if tensors[key].ndim == 3 and tensors[key].shape[0] == 1:
                tensors[key] = tensors[key].squeeze(0)
    
    # 创建图像和网格布局（3行3列），新增gt和prediction的位置
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(2, 5, figure=fig, wspace=0.3, hspace=0.4)
    
    # 定义每个张量的可视化配置
    visual_configs = [
        {
            'tensor': tensors['gt'],
            'title': 'Ground Truth',
            'cmap': None,  # 图像使用原始颜色，不使用颜色映射
            'vmin': 0,
            'vmax': 1
        },
        {
            'tensor': tensors['prediction'],
            'title': 'Prediction',
            'cmap': None,  # 图像使用原始颜色，不使用颜色映射
            'vmin': 0,
            'vmax': 1
        },
        {
            'tensor': tensors['uncertainty'],
            'title': 'Uncertainty: dino feature of gt',
            'cmap': 'viridis',
            'vmin': 0,
            'vmax': 1
        },
        {
            'tensor': tensors['dino_cosine'],
            'title': 'dino_cosine',
            'cmap': 'coolwarm',
            'vmin': -1,
            'vmax': 1
        },
        {
            'tensor': tensors['dino_part'],
            'title': 'dino_part: dino_cosine to [0, 1]',
            'cmap': 'viridis',
            'vmin': 0,
            'vmax': None
        },
        {
            'tensor': tensors['uncertainty_loss'],
            'title': 'uncertainty_loss = dino_part * dino_downsample',
            'cmap': 'magma',
            'vmin': 0,         # 损失范围设为0到动态最大值（自动计算）
            'vmax': None       # 若需固定最大值，可设为如vmax=5
        },
        {
            'tensor': tensors['loss_mult1'],
            'title': 'loss_mult1 = 1/(2*uncertainty.pow(2))',
            'cmap': 'inferno',
            'vmin': 0,
            'vmax': 3
        },
        {
            'tensor': tensors['loss_mult2'],
            'title': 'loss_mult max is 3',
            'cmap': 'plasma',
            'vmin': 0,
            'vmax': 3
        },
        {
            'tensor': tensors['loss_mult3'],
            'title': '<mean get mask',
            'cmap': 'binary',  # 二值化结果使用binary映射
            'vmin': 0,
            'vmax': 1
        },
        {
            'tensor': tensors['loss_mult4'],
            'title': 'warmup in loss_mult3',
            'cmap': 'viridis',  # 二值化结果使用binary映射
            'vmin': 0,
            'vmax': 1
        }
    ]
    
    # 绘制每个张量并统一颜色映射
    for i, config in enumerate(visual_configs):
        ax = fig.add_subplot(gs[i])
        # 计算uncertainty_loss的动态vmax（若未指定）
        if config['vmax'] is None and config['title'] == 'Uncertainty: dino feature of gt':
            config['vmax'] = np.percentile(config['tensor'], 99)  # 取99%分位数作为最大值，避免异常值影响
        im = ax.imshow(config['tensor'], cmap=config['cmap'], 
                      vmin=config['vmin'], vmax=config['vmax'])
        ax.set_title(config['title'], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # 添加颜色条（每个子图独立，确保数值与颜色一一对应）
        cbar = fig.colorbar(im, ax=ax, pad=0.02, aspect=20)
        cbar.ax.tick_params(labelsize=8)  # 调整颜色条标签字体大小
    
    plt.tight_layout(pad=0.5)
    

    # 保存和显示图像
    if save_path:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig

def visualize_all_tensors2(
    uncertainty, 
    dino_cosine, 
    dino_part, 
    uncertainty_loss, 
    loss_mult1, 
    loss_mult2, 
    save_path=None, 
    show=True
):
    """
    可视化多个张量并拼接为单张图片（数值到颜色的映射一致）
    
    参数:
        uncertainty: 不确定性张量（建议范围：0到1）
        dino_cosine: DINO余弦相似度张量（范围：-1到1）
        dino_part: 处理后的DINO相似度张量（范围：0到1）
        uncertainty_loss: 不确定性损失张量（建议范围：0到最大值）
        loss_mult1: 损失乘数1（范围：0到3）
        loss_mult2: 损失乘数2（范围：0到3）
        save_path: 保存路径，若为None则不保存
        show: 是否显示图像
    """
    # 确保所有张量在CPU上并转换为numpy数组
    tensors = {
        'uncertainty': uncertainty.cpu().numpy() if isinstance(uncertainty, torch.Tensor) else uncertainty,
        'dino_cosine': dino_cosine.cpu().numpy() if isinstance(dino_cosine, torch.Tensor) else dino_cosine,
        'dino_part': dino_part.cpu().numpy() if isinstance(dino_part, torch.Tensor) else dino_part,
        'uncertainty_loss': uncertainty_loss.cpu().numpy() if isinstance(uncertainty_loss, torch.Tensor) else uncertainty_loss,
        'loss_mult1': loss_mult1.cpu().numpy() if isinstance(loss_mult1, torch.Tensor) else loss_mult1,
        'loss_mult2': loss_mult2.cpu().numpy() if isinstance(loss_mult2, torch.Tensor) else loss_mult2
    }
    
    # 确保所有张量是2D（若为3D则取第一个通道）
    for key in tensors:
        if tensors[key].ndim == 3 and tensors[key].shape[0] == 1:
            tensors[key] = tensors[key].squeeze(0)
    
    # 创建图像和网格布局（1行6列）
    fig = plt.figure(figsize=(18, 3))
    gs = GridSpec(1, 6, figure=fig, wspace=0.3, hspace=0.4)
    
    # 定义每个张量的可视化配置（固定vmin和vmax）
    visual_configs = [
        {
            'tensor': tensors['uncertainty'],
            'title': 'Uncertainty',
            'cmap': 'viridis',
            'vmin': 0,         # 不确定性范围设为0到1（可根据实际数据调整）
            'vmax': 1
        },
        {
            'tensor': tensors['dino_cosine'],
            'title': 'DINO Cosine Similarity',
            'cmap': 'coolwarm',
            'vmin': -1,        # 余弦相似度固定为-1到1
            'vmax': 1
        },
        {
            'tensor': tensors['dino_part'],
            'title': 'Processed DINO Similarity',
            'cmap': 'viridis',
            'vmin': 0,         # 处理后的相似度固定为0到1
            'vmax': 1
        },
        {
            'tensor': tensors['uncertainty_loss'],
            'title': 'Uncertainty Loss',
            'cmap': 'magma',
            'vmin': 0,         # 损失范围设为0到动态最大值（自动计算）
            'vmax': None       # 若需固定最大值，可设为如vmax=5
        },
        {
            'tensor': tensors['loss_mult1'],
            'title': 'Loss Multiplier 1',
            'cmap': 'inferno',
            'vmin': 0,         # 损失乘数固定为0到3
            'vmax': 3
        },
        {
            'tensor': tensors['loss_mult2'],
            'title': 'Loss Multiplier 2',
            'cmap': 'plasma',
            'vmin': 0,         # 损失乘数固定为0到3
            'vmax': 3
        }
    ]
    
    # 绘制每个张量并统一颜色映射
    for i, config in enumerate(visual_configs):
        ax = fig.add_subplot(gs[i])
        # 计算uncertainty_loss的动态vmax（若未指定）
        if config['vmax'] is None and config['title'] == 'Uncertainty Loss':
            config['vmax'] = np.percentile(config['tensor'], 99)  # 取99%分位数作为最大值，避免异常值影响
        im = ax.imshow(config['tensor'], cmap=config['cmap'], 
                      vmin=config['vmin'], vmax=config['vmax'])
        ax.set_title(config['title'], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # 添加颜色条（每个子图独立，确保数值与颜色一一对应）
        cbar = fig.colorbar(im, ax=ax, pad=0.02, aspect=20)
        cbar.ax.tick_params(labelsize=8)  # 调整颜色条标签字体大小
    
    plt.tight_layout(pad=0.5)  # 调整整体布局间距
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图像
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig
def visualize_all_tensors(
    uncertainty, 
    dino_cosine, 
    dino_part, 
    uncertainty_loss, 
    loss_mult1, 
    loss_mult2, 
    save_path=None, 
    show=True
):
    """
    可视化多个张量并拼接为单张图片
    
    参数:
        uncertainty: 不确定性张量
        dino_cosine: DINO余弦相似度张量
        dino_part: 处理后的DINO相似度张量
        uncertainty_loss: 不确定性损失张量
        loss_mult1: 损失乘数1
        loss_mult2: 损失乘数2
        save_path: 保存路径，若为None则不保存
        show: 是否显示图像
    """
    # 确保所有张量在CPU上并转换为numpy数组
    tensors = {
        'uncertainty': uncertainty.cpu().numpy() if isinstance(uncertainty, torch.Tensor) else uncertainty,
        'dino_cosine': dino_cosine.cpu().numpy() if isinstance(dino_cosine, torch.Tensor) else dino_cosine,
        'dino_part': dino_part.cpu().numpy() if isinstance(dino_part, torch.Tensor) else dino_part,
        'uncertainty_loss': uncertainty_loss.cpu().numpy() if isinstance(uncertainty_loss, torch.Tensor) else uncertainty_loss,
        'loss_mult1': loss_mult1.cpu().numpy() if isinstance(loss_mult1, torch.Tensor) else loss_mult1,
        'loss_mult2': loss_mult2.cpu().numpy() if isinstance(loss_mult2, torch.Tensor) else loss_mult2
    }
    
    # 确保所有张量是2D（若为3D则取第一个通道）
    for key in tensors:
        if tensors[key].ndim == 3 and tensors[key].shape[0] == 1:
            tensors[key] = tensors[key].squeeze(0)
    
    # 创建图像和网格布局（3行2列）
    fig = plt.figure(figsize=(12, 18))
    gs = GridSpec(1, 6, figure=fig, wspace=0.3, hspace=0.4)
    
    # 定义每个张量的可视化配置
    visual_configs = [
        {'tensor': tensors['uncertainty'], 'title': 'Uncertainty', 'cmap': 'viridis', 'vmin': None, 'vmax': None},
        {'tensor': tensors['dino_cosine'], 'title': 'DINO Cosine Similarity', 'cmap': 'coolwarm', 'vmin': -1, 'vmax': 1},
        {'tensor': tensors['dino_part'], 'title': 'Processed DINO Similarity', 'cmap': 'viridis', 'vmin': 0, 'vmax': 1},
        {'tensor': tensors['uncertainty_loss'], 'title': 'Uncertainty Loss', 'cmap': 'magma', 'vmin': 0, 'vmax': None},
        {'tensor': tensors['loss_mult1'], 'title': 'Loss Multiplier 1', 'cmap': 'inferno', 'vmin': 0, 'vmax': 3},  # 假设loss_mult1最大值为3
        {'tensor': tensors['loss_mult2'], 'title': 'Loss Multiplier 2', 'cmap': 'plasma', 'vmin': 0, 'vmax': 3}   # 假设loss_mult2最大值为3
    ]
    
    # 绘制每个张量
    for i, config in enumerate(visual_configs):
        ax = fig.add_subplot(gs[i//2, i%2])
        im = ax.imshow(config['tensor'], cmap=config['cmap'], 
                      vmin=config['vmin'], vmax=config['vmax'])
        ax.set_title(config['title'], fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, pad=0.01)  # 添加颜色条
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图像
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig
def visualize_dino_cosine(dino_cosine, dino_part, save_path=None, show=True):
        """
        可视化 DINO 余弦相似度和处理后的 dino_part
        
        参数:
            dino_cosine: 原始余弦相似度张量 (范围 [-1, 1])
            dino_part: 处理后的相似度张量 (范围 [0, 1])
            save_path: 保存路径，若为None则不保存
            show: 是否显示图像
        """
        # 确保输入是 numpy 数组
        if isinstance(dino_cosine, torch.Tensor):
            dino_cosine = dino_cosine.cpu().numpy()
        
        if isinstance(dino_part, torch.Tensor):
            dino_part = dino_part.cpu().numpy()
        
        # 创建图像显示布局
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 可视化原始余弦相似度 (-1 到 1)
        im1 = axes[0].imshow(dino_cosine, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0].set_title('DINO Cosine Similarity (-1 to 1)')
        fig.colorbar(im1, ax=axes[0])
        
        # 可视化处理后的 dino_part (0 到 1)
        im2 = axes[1].imshow(dino_part, cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('Processed DINO Similarity (0 to 1)')
        fig.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 显示图像
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig


class UncertaintyModel(nn.Module):
    img_norm_mean: Tensor
    img_norm_std: Tensor

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.backbone = getattr(dinov2, config.uncertainty_backbone)(pretrained=True)
        self.patch_size = self.backbone.patch_size
        in_features = self.backbone.embed_dim
        self.conv_seg = nn.Conv2d(in_features, 1, kernel_size=1)
        self.bn = nn.SyncBatchNorm(in_features)
        nn.init.normal_(self.conv_seg.weight.data, 0, 0.01)
        nn.init.zeros_(assert_not_none(self.conv_seg.bias).data)

        img_norm_mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
        img_norm_std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)
        self.register_buffer("img_norm_mean", img_norm_mean / 255.0)
        self.register_buffer("img_norm_std", img_norm_std / 255.0)

        self._images_cache = {}

        # Freeze dinov2 backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _get_pad(self, size):
        new_size = math.ceil(size / self.patch_size) * self.patch_size
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    def _initialize_head_from_checkpoint(self):
        # ADA20 classes to ignore
        cls_to_ignore = [13, 21, 81, 84]
        # Pull the checkpoint
        backbone = self.config.uncertainty_backbone
        url = f"https://dl.fbaipublicfiles.com/dinov2/{backbone}/{backbone}_ade20k_linear_head.pth"
        with urllib.request.urlopen(url) as f:
            checkpoint_data = f.read()
        checkpoint = torch.load(io.BytesIO(checkpoint_data), map_location="cpu")
        old_weight = checkpoint["state_dict"]["decode_head.conv_seg.weight"]
        new_weight = torch.empty(1, old_weight.shape[1], 1, 1)
        nn.init.normal_(new_weight, 0, 0.0001)
        new_weight[:, cls_to_ignore] = old_weight[:, cls_to_ignore] * 1000
        nn.init.zeros_(assert_not_none(self.conv_seg.bias).data)
        self.conv_seg.weight.data.copy_(new_weight)

        # Load the bn data
        self.bn.load_state_dict(
            {
                k[len("decode_head.bn.") :]: v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("decode_head.bn.")
            }
        )

    def _get_dino_cached(self, x, cache_entry=None):
        if cache_entry is None or (cache_entry, x.shape) not in self._images_cache:
            with torch.no_grad():
                x = self.backbone.get_intermediate_layers(
                    x, n=[self.backbone.num_heads - 1], reshape=True
                )[-1]
            if cache_entry is not None:
                self._images_cache[(cache_entry, x.shape)] = x.detach().cpu()
        else:
            x = self._images_cache[(cache_entry, x.shape)].to(x.device)
        return x

    def _compute_cosine_similarity(
        self, x, y, _x_cache=None, _y_cache=None, max_size=None
    ):
        # Normalize data
        h, w = x.shape[2:]
        if max_size is not None and (max_size < h or max_size < w):
            assert max_size % 14 == 0, "max_size must be divisible by 14"
            scale_factor = min(max_size / x.shape[-2], max_size / x.shape[-1])
            nh = int(h * scale_factor)
            nw = int(w * scale_factor)
            nh = ((nh + 13) // 14) * 14
            nw = ((nw + 13) // 14) * 14
            x = F.interpolate(x, size=(nh, nw), mode="bilinear")
            y = F.interpolate(y, size=(nh, nw), mode="bilinear")

        x = (x - self.img_norm_mean[None, :, None, None]) / self.img_norm_std[
            None, :, None, None
        ]
        y = (y - self.img_norm_mean[None, :, None, None]) / self.img_norm_std[
            None, :, None, None
        ]
        pads = list(
            itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1])
        )
        x = F.pad(x, pads)
        padded_shape = x.shape
        y = F.pad(y, pads)

        # 提取输入图像的 DINO 特征
        with torch.no_grad():
            x = self._get_dino_cached(x, _x_cache)
            y = self._get_dino_cached(y, _y_cache)

        cosine = F.cosine_similarity(x, y, dim=1).unsqueeze(1)
        cosine: Tensor = F.interpolate(
            cosine, size=padded_shape[2:], mode="bilinear", align_corners=False
        )

        # Remove padding
        cosine = cosine[:, :, pads[2] : h + pads[2], pads[0] : w + pads[0]]
        if max_size is not None and (max_size < h or max_size < w):
            cosine = F.interpolate(
                cosine, size=(h, w), mode="bilinear", align_corners=False
            )
        return cosine.squeeze(1)

    def _forward_uncertainty_features(
        self, inputs: Tensor, _cache_entry=None
    ) -> Tensor:
        # Normalize data
        inputs = (inputs - self.img_norm_mean[None, :, None, None]) / self.img_norm_std[
            None, :, None, None
        ]
        h, w = inputs.shape[2:]
        pads = list(
            itertools.chain.from_iterable(self._get_pad(m) for m in inputs.shape[:1:-1])
        )
        inputs = F.pad(inputs, pads)

        x = self._get_dino_cached(inputs, _cache_entry)

        x = F.dropout2d(x, p=self.config.uncertainty_dropout, training=self.training)
        x = self.bn(x)
        logits = self.conv_seg(x)
        # We could also do this using weight init,
        # but we want to have a prior then doing L2 regularization
        logits = logits + math.log(math.exp(1) - 1)

        # Rescale to input size
        logits = F.softplus(logits)
        logits: Tensor = F.interpolate(
            logits, size=inputs.shape[2:], mode="bilinear", align_corners=False
        )
        logits = logits.clamp(min=self.config.uncertainty_clip_min)

        # Add padding
        logits = logits[:, :, pads[2] : h + pads[2], pads[0] : w + pads[0]]
        return logits

    @property
    def device(self):
        return self.img_norm_mean.device

    def forward(self, image: Tensor, _cache_entry=None):
        return self._forward_uncertainty_features(image, _cache_entry=_cache_entry)

    def setup_data(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def _load_image(self, img):
        return torch.from_numpy(
            np.moveaxis(convert_image_dtype(img, np.float32), -1, 0)[None]
        ).to(self.device)

    def _scale_input(self, x, max_size: Optional[int] = 504):
        h, w = nh, nw = x.shape[2:]
        if max_size is not None:
            scale_factor = min(max_size / x.shape[-2], max_size / x.shape[-1])
            if scale_factor >= 1:
                return x
            nw = int(w * scale_factor)
            nh = int(h * scale_factor)
            nh = ((nh + 13) // 14) * 14
            nw = ((nw + 13) // 14) * 14
            x = F.interpolate(x, size=(nh, nw), mode="bilinear")
        return x

    def _dino_plus_ssim(self, gt, prediction, _cache_entry=None, max_size=None):
        gt_down = dino_downsample(gt, max_size=max_size)
        prediction_down = dino_downsample(prediction, max_size=max_size)
        dino_cosine = self._compute_cosine_similarity(
            gt_down, prediction_down, _x_cache=_cache_entry
        ).detach()
        dino_part = (1 - dino_cosine.sub(0.5).div(0.5)).clip(0.0, 1)
        msssim_part = (
            1 - msssim(gt_down, prediction_down, min_size=80).unsqueeze(1).detach()
        )
        return torch.min(dino_part, msssim_part)



    def _compute_losses(self, model_path, iteration, gt, prediction, prefix="", _cache_entry=None):
        uncertainty = self(
            self._scale_input(gt, self.config.uncertainty_dino_max_size),
            _cache_entry=_cache_entry,
        )
        log_uncertainty = torch.log(uncertainty)
        # _dssim_go = dssim_go(gt, prediction, size_average=False).unsqueeze(1).clamp_max(self.config.uncertainty_dssim_clip_max)
        # _dssim_go = 1 - ssim(gt, prediction).unsqueeze(1)
        _ssim = ssim_down(gt, prediction, max_size=400).unsqueeze(1)
        _msssim = msssim(gt, prediction, max_size=400, min_size=80).unsqueeze(1)

        if iteration==1:
            print(f'#'*30)
            print(self.config.uncertainty_mode)

        if self.config.uncertainty_mode == "l2reg":
            if uncertainty.shape[2:] != gt.shape[2:]:
                uncertainty = F.interpolate(
                    uncertainty, size=gt.shape[2:], mode="bilinear", align_corners=False
                )
            loss_mult = 1 / (2 * uncertainty.pow(2))
            uncertainty_loss = (1 - _msssim.detach()) * loss_mult
        elif self.config.uncertainty_mode == "l1reg":
            if uncertainty.shape[2:] != gt.shape[2:]:
                uncertainty = F.interpolate(
                    uncertainty, size=gt.shape[2:], mode="bilinear", align_corners=False
                )
            loss_mult = 1 / uncertainty
            uncertainty_loss = (1 - _msssim.detach()) * loss_mult
        elif self.config.uncertainty_mode == "dino":
            # jqq loss_mult
            # loss_mult = 1 / (2 * uncertainty.pow(2)) # 平方反比，适用于需要强烈抑制高不确定性区域的场景。
            # loss_mult = 1 / uncertainty # 线性反比，适用于希望平衡高 / 低不确定性区域的场景，避免对高不确定性区域过度忽略
            
            # Compute dino loss
            
            loss_mult = 1 / (2 * uncertainty.pow(2))
            gt_down = dino_downsample(gt, max_size=350)
            prediction_down = dino_downsample(prediction, max_size=350)
            
            loss_mult1=loss_mult
            # 计算 DINO 特征的余弦相似度
            dino_cosine = self._compute_cosine_similarity(
                gt_down, prediction_down, _x_cache=_cache_entry
            ).detach()

            # 将余弦相似度映射到 [0, 1] 范围内
            dino_part = (1 - dino_cosine.sub(0.5).div(0.5)).clip(0.0, 1)

            # 将 dino_part 与 loss_mult 相乘，得到最终的损失
            uncertainty_loss = dino_part * dino_downsample(loss_mult, max_size=350)
            if loss_mult.shape[2:] != gt.shape[2:]:
                loss_mult = F.interpolate(
                    loss_mult, size=gt.shape[2:], mode="bilinear", align_corners=False
                )
            # 将变量 loss_mult 的值限制在最大值为 3
            loss_mult = loss_mult.clamp_max(3)

            loss_mult2=loss_mult

            # 计算loss_mult2的平均值
            loss_mult_mean = loss_mult2.mean()
            # 将大于平均值的元素设置为True，其余为False
            loss_mult3 = (loss_mult2 > loss_mult_mean).to(dtype=loss_mult.dtype)

            # 可视化所有张量
            if iteration % 100 == 0:  # 每100次迭代可视化一次
                
                uncertainty_warmup_iters = 1300
                uncertainty_warmup_start = 700
                if iteration < uncertainty_warmup_start:
                    loss_mult4 = torch.ones_like(loss_mult1)  # warmup前 实际*的 loss_mult
                elif iteration < uncertainty_warmup_start + uncertainty_warmup_iters:
                    # p = (iteration - uncertainty_warmup_start) / uncertainty_warmup_iters
                    # loss_mult4 = 1 + p * (loss_mult - 1) # 轮次约束的loss_mult

                    # jqq
                    p = (iteration - uncertainty_warmup_start) / uncertainty_warmup_iters
                    # 使用sigmoid函数实现平滑过渡 (更平滑的S形曲线)
                    # 这里使用5作为陡度参数，可以根据需要调整
                    smooth_p = torch.sigmoid(torch.tensor(5 * (p - 0.5)))
                    loss_mult4=torch.ones_like(loss_mult3) * (1 - smooth_p) + loss_mult3 * smooth_p
                else:
                    loss_mult4=loss_mult3

                # 确保所有张量在CPU上
                gt_cpu = gt.cpu() # gt
                prediction_cpu = prediction.cpu() # render
                uncertainty_cpu = uncertainty.cpu() # dino提取的gt的特征
                dino_cosine_cpu = dino_cosine.cpu() # DINO 特征的余弦相似度
                dino_part_cpu = dino_part.cpu() # 将余弦相似度映射到 [0, 1] 范围内
                uncertainty_loss_cpu = uncertainty_loss.cpu() # uncertainty_loss = dino_part * dino_downsample
                loss_mult1_cpu = loss_mult1.cpu() # loss_mult1 = 1 / (2 * uncertainty.pow(2))
                loss_mult2_cpu = loss_mult2.cpu() # loss_mult 的值限制在最大值为 3
                loss_mult3_cpu = loss_mult3.cpu() # 以平均值为阈值，得到二值mask
                loss_mult4_cpu = loss_mult4.cpu()

                # 仅在 loss_mult4 存在时传递该参数
                visualize_kwargs = {
                    'gt': gt_cpu.squeeze(0).detach(),
                    'prediction': prediction_cpu.squeeze(0).detach(),
                    'uncertainty': uncertainty_cpu.squeeze(0).detach(),
                    'dino_cosine': dino_cosine_cpu.squeeze(0).detach(),
                    'dino_part': dino_part_cpu.squeeze(0).detach(),
                    'uncertainty_loss': uncertainty_loss_cpu.squeeze(0).detach(),
                    'loss_mult1': loss_mult1_cpu.squeeze(0).detach(),
                    'loss_mult2': loss_mult2_cpu.squeeze(0).detach(),
                    'loss_mult3': loss_mult3_cpu.squeeze(0).detach(),
                    'loss_mult4': loss_mult4_cpu.squeeze(0).detach(),
                }
                    
                visualize_all_tensors3(**visualize_kwargs, save_path=f"{model_path}/mid_result/all_tensors_iter_{iteration}.png")

        elif self.config.uncertainty_mode == "dino+mssim":
            loss_mult = 1 / (2 * uncertainty.pow(2))
            gt_down = dino_downsample(gt, max_size=350)
            prediction_down = dino_downsample(prediction, max_size=350)
            dino_cosine = self._compute_cosine_similarity(
                gt_down, prediction_down, _x_cache=_cache_entry
            ).detach()
            dino_part = (1 - dino_cosine.sub(0.5).div(0.5)).clip(0.0, 1)
            msssim_part = (
                1 - msssim(gt_down, prediction_down, min_size=80).unsqueeze(1).detach()
            )
            uncertainty_loss = torch.min(dino_part, msssim_part) * dino_downsample(
                loss_mult, max_size=350
            )
            if loss_mult.shape[2:] != gt.shape[2:]:
                loss_mult = F.interpolate(
                    loss_mult, size=gt.shape[2:], mode="bilinear", align_corners=False
                )
            loss_mult = loss_mult.clamp_max(3)

        else:
            raise ValueError(
                f"Invalid uncertainty_mode: {self.config.uncertainty_mode}"
            )

        beta = log_uncertainty.mean()
        loss = (
            uncertainty_loss.mean() + self.config.uncertainty_regularizer_weight * beta
        )

        ssim_discounted = (_ssim * loss_mult).sum() / loss_mult.sum()
        mse = torch.pow(gt - prediction, 2)
        mse_discounted = (mse * loss_mult).sum() / loss_mult.sum()
        psnr_discounted = 10 * torch.log10(1 / mse_discounted)

        metrics = {
            f"{prefix}loss": loss.item(),
            f"{prefix}ssim": _ssim.mean().item(),
            f"{prefix}msssim": _msssim.mean().item(),
            f"{prefix}ssim_discounted": ssim_discounted.item(),
            f"{prefix}mse_discounted": mse_discounted.item(),
            f"{prefix}psnr_discounted": psnr_discounted.item(),
            f"{prefix}beta": beta.item(),
        }
        return uncertainty_loss, loss, metrics, loss_mult.detach()

    def get_loss(self, model_path, iteration, gt_image, image, prefix="", _cache_entry=None):
        gt_torch = gt_image.unsqueeze(0)
        image = image.unsqueeze(0)
        _, loss, metrics, loss_mult = self._compute_losses(
            model_path, iteration, gt_torch, image, prefix, _cache_entry=_cache_entry
        )
        loss_mult = loss_mult.squeeze(0)
        metrics[f"{prefix}uncertainty_loss"] = metrics.pop(f"{prefix}loss")
        metrics.pop(f"{prefix}ssim")
        return loss, metrics, loss_mult
    

    def qq_compute_losses(self, iteration, gt, prediction, prefix="", _cache_entry=None):
        uncertainty = self(
            self._scale_input(gt, self.config.uncertainty_dino_max_size),
            _cache_entry=_cache_entry,
        )
        log_uncertainty = torch.log(uncertainty)
        # _dssim_go = dssim_go(gt, prediction, size_average=False).unsqueeze(1).clamp_max(self.config.uncertainty_dssim_clip_max)
        # _dssim_go = 1 - ssim(gt, prediction).unsqueeze(1)
        _ssim = ssim_down(gt, prediction, max_size=400).unsqueeze(1)
        _msssim = msssim(gt, prediction, max_size=400, min_size=80).unsqueeze(1)

        if iteration==1:
            print(f'#'*30)
            print(self.config.uncertainty_mode)

        if self.config.uncertainty_mode == "l2reg":
            if uncertainty.shape[2:] != gt.shape[2:]:
                uncertainty = F.interpolate(
                    uncertainty, size=gt.shape[2:], mode="bilinear", align_corners=False
                )
            loss_mult = 1 / (2 * uncertainty.pow(2))
            uncertainty_loss = (1 - _msssim.detach()) * loss_mult
        elif self.config.uncertainty_mode == "l1reg":
            if uncertainty.shape[2:] != gt.shape[2:]:
                uncertainty = F.interpolate(
                    uncertainty, size=gt.shape[2:], mode="bilinear", align_corners=False
                )
            loss_mult = 1 / uncertainty
            uncertainty_loss = (1 - _msssim.detach()) * loss_mult
        elif self.config.uncertainty_mode == "dino":
            # jqq loss_mult
            # loss_mult = 1 / (2 * uncertainty.pow(2)) # 平方反比，适用于需要强烈抑制高不确定性区域的场景。
            # loss_mult = 1 / uncertainty # 线性反比，适用于希望平衡高 / 低不确定性区域的场景，避免对高不确定性区域过度忽略
            
            # Compute dino loss
            
            loss_mult = 1 / (2 * uncertainty.pow(2))
            gt_down = dino_downsample(gt, max_size=350)
            prediction_down = dino_downsample(prediction, max_size=350)
            
            loss_mult1=loss_mult
            # 计算 DINO 特征的余弦相似度
            dino_cosine = self._compute_cosine_similarity(
                gt_down, prediction_down, _x_cache=_cache_entry
            ).detach()

            # 将余弦相似度映射到 [0, 1] 范围内
            dino_part = (1 - dino_cosine.sub(0.5).div(0.5)).clip(0.0, 1)

            # 将 dino_part 与 loss_mult 相乘，得到最终的损失
            uncertainty_loss = dino_part * dino_downsample(loss_mult, max_size=350)
            if loss_mult.shape[2:] != gt.shape[2:]:
                loss_mult = F.interpolate(
                    loss_mult, size=gt.shape[2:], mode="bilinear", align_corners=False
                )
            # 将变量 loss_mult 的值限制在最大值为 3
            loss_mult = loss_mult.clamp_max(3)

            loss_mult2=loss_mult

        elif self.config.uncertainty_mode == "dino+mssim":
            loss_mult = 1 / (2 * uncertainty.pow(2))
            gt_down = dino_downsample(gt, max_size=350)
            prediction_down = dino_downsample(prediction, max_size=350)
            dino_cosine = self._compute_cosine_similarity(
                gt_down, prediction_down, _x_cache=_cache_entry
            ).detach()
            dino_part = (1 - dino_cosine.sub(0.5).div(0.5)).clip(0.0, 1)
            msssim_part = (
                1 - msssim(gt_down, prediction_down, min_size=80).unsqueeze(1).detach()
            )
            uncertainty_loss = torch.min(dino_part, msssim_part) * dino_downsample(
                loss_mult, max_size=350
            )
            if loss_mult.shape[2:] != gt.shape[2:]:
                loss_mult = F.interpolate(
                    loss_mult, size=gt.shape[2:], mode="bilinear", align_corners=False
                )
            loss_mult = loss_mult.clamp_max(3)

        else:
            raise ValueError(
                f"Invalid uncertainty_mode: {self.config.uncertainty_mode}"
            )

        beta = log_uncertainty.mean()
        loss = (
            uncertainty_loss.mean() + self.config.uncertainty_regularizer_weight * beta
        )

        ssim_discounted = (_ssim * loss_mult).sum() / loss_mult.sum()
        mse = torch.pow(gt - prediction, 2)
        mse_discounted = (mse * loss_mult).sum() / loss_mult.sum()
        psnr_discounted = 10 * torch.log10(1 / mse_discounted)

        metrics = {
            f"{prefix}loss": loss.item(),
            f"{prefix}ssim": _ssim.mean().item(),
            f"{prefix}msssim": _msssim.mean().item(),
            f"{prefix}ssim_discounted": ssim_discounted.item(),
            f"{prefix}mse_discounted": mse_discounted.item(),
            f"{prefix}psnr_discounted": psnr_discounted.item(),
            f"{prefix}beta": beta.item(),
        }

        return uncertainty.detach(), dino_cosine.detach(), dino_part.detach(), loss_mult1.detach(), loss_mult2.detach(), uncertainty_loss, loss, metrics, loss_mult.detach()
    
    def qq_get_loss(self, iteration, gt_image, image, prefix="", _cache_entry=None):
        gt_torch = gt_image.unsqueeze(0)
        image = image.unsqueeze(0)
        uncertainty, dino_cosine, dino_part, loss_mult1, loss_mult2, uncertainty_loss, loss, metrics, loss_mult = self.qq_compute_losses(
            iteration, gt_torch, image, prefix, _cache_entry=_cache_entry
        )
        loss_mult = loss_mult.squeeze(0)
        metrics[f"{prefix}uncertainty_loss"] = metrics.pop(f"{prefix}loss")
        metrics.pop(f"{prefix}ssim")
        return uncertainty, dino_cosine, dino_part, loss_mult1, loss_mult2, uncertainty_loss, loss, metrics, loss_mult

    @staticmethod
    def load(path: str):
        ckpt = torch.load(os.path.join(path, "checkpoint.pth"), map_location="cpu")
        config = OmegaConf.structured(Config)
        config = cast(
            Config, OmegaConf.merge(config, OmegaConf.create(ckpt.pop("config")))
        )
        model = UncertaintyModel(config)
        model.load_state_dict(ckpt, strict=False)
        return model

    def save(self, path: str):
        state = self.state_dict()
        state["config"] = OmegaConf.to_yaml(self.config, resolve=True)
        torch.save(state, os.path.join(path, "checkpoint.pth"))
