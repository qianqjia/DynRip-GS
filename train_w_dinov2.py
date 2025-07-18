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

import os
from matplotlib import pyplot as plt
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene
from scene.gaussian_model_dinov2 import GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments.args_qq import ModelParams, PipelineParams, OptimizationParams
import math
import torch.nn.functional as F
from scene.uncertainty_model.model import Config 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image
from matplotlib.gridspec import GridSpec

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='alex').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')

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
            if tensors[key].ndim == 0:  # 处理0维标量情况
                tensors[key] = tensors[key].reshape(1, 1)  # 转换为2D张量
            elif tensors[key].ndim == 3 and tensors[key].shape[0] == 1:
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


# SSIM
def ssim(img1, img2, window_size=11, size_average=True):
    sigma = 1.5
    channel = img1.size(-3)
    # Create window
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(-3)


def scale_grads(values, scale):
    grad_values = values * scale
    rest_values = values.detach() * (1 - scale)
    return grad_values + rest_values


def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    config = Config()
    for key, value in opt.__dict__.items():
        if hasattr(config, key):
            # print(key, value)
            setattr(config, key, value)
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, uncertainty_config=config)
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)


    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", dynamic_ncols=True, ascii=True)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        # network gui not available in scaffold-gs yet
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        gt_image = viewpoint_cam.original_image.cuda()


        # uncertainty_loss, metrics, loss_mult = gaussians.uncertainty_model.get_loss(scene.model_path, iteration, gt_image, image.detach(), _cache_entry=('train', viewpoint_cam.colmap_id))
        
        uncertainty, dino_cosine, dino_part, loss_mult1, loss_mult2, uncertainty_loss, loss, metrics, loss_mult= gaussians.uncertainty_model.qq_get_loss(iteration, gt_image, image.detach(), _cache_entry=('train', viewpoint_cam.colmap_id))


        # 确保 uncertainty_loss 是一个标量
        if uncertainty_loss.dim() > 0:
            uncertainty_loss = uncertainty_loss.mean()
        
        # jqq
            
        # BVI数据集
        loss_mult = (loss_mult > 0.5).to(dtype=loss_mult.dtype)
            
        # IW数据集
        # 计算loss_mult的平均值
        loss_mult_mean = loss_mult.mean()
        # 将大于平均值的元素设置为True，其余为False
        loss_mult = (loss_mult > loss_mult_mean).to(dtype=loss_mult.dtype)

        loss_mult3=loss_mult # 二值 loss_mult

        if iteration < opt.uncertainty_warmup_start:
            loss_mult = 1
            loss_mult4 = torch.ones_like(loss_mult1)  # warmup前 实际*的 loss_mult
        elif iteration < opt.uncertainty_warmup_start + opt.uncertainty_warmup_iters:
            # p = (iteration - uncertainty_warmup_start) / uncertainty_warmup_iters
            # loss_mult4 = 1 + p * (loss_mult - 1) # 轮次约束的loss_mult

            # jqq
            p = (iteration - opt.uncertainty_warmup_start) / opt.uncertainty_warmup_iters
            # loss_mult = 1 + p * (loss_mult - 1) # 轮次约束的loss_mult

            # jqq
            # 使用sigmoid函数实现平滑过渡 (更平滑的S形曲线)
            # 这里使用5作为陡度参数，可以根据需要调整
            smooth_p = torch.sigmoid(torch.tensor(5 * (p - 0.5)))
            torch.ones_like(loss_mult3) * (1 - smooth_p) + loss_mult3 * smooth_p
            loss_mult = torch.ones_like(loss_mult3) * (1 - smooth_p) + loss_mult3 * smooth_p
            loss_mult4 = loss_mult
        else:
            loss_mult4=loss_mult3


        if opt.uncertainty_center_mult:
            loss_mult = loss_mult.sub(loss_mult.mean() - 1).clamp(0, 2)
        if opt.uncertainty_scale_grad:
            image = scale_grads(image, loss_mult)
            image_toned = scale_grads(image_toned, loss_mult)
            loss_mult = 1



        
        Ll1 = torch.nn.functional.l1_loss(image, gt_image, reduction='none')

        ssim_loss = (1.0 - ssim(image, gt_image, size_average=False))
        scaling_reg = scaling.prod(dim=1).mean()

        # # Detach uncertainty loss if in protected iter after opacity reset
        # last_densify_iter = min(iteration, opt.update_until - 1)
        # # last_dentify_iter = (last_densify_iter // opt.opacity_reset_interval) * opt.opacity_reset_interval
        # if iteration < last_densify_iter + opt.uncertainty_protected_iters:
        #     # Keep track of max radii in image-space for pruning
        #     try:
        #         uncertainty_loss = uncertainty_loss.detach()  # type: ignore
        #     except AttributeError:
        #         pass

        # loss = (1.0 - opt.lambda_dssim) * (Ll1 * loss_mult).mean() + opt.lambda_dssim * (ssim_loss * loss_mult).mean() + uncertainty_loss + 0.01*scaling_reg

        # 现在*的是 warmup loss_mult
         # 将loss_mult4调整为Ll1的大小
        loss_mult4 = (dino_part < 0.9).to(dtype=dino_part.dtype)
        
        # 确保loss_mult4是[B, 1, H', W']格式
        loss_mult4_expanded = loss_mult4.unsqueeze(1)
        
        # 插值到[540, 960]尺寸
        loss_mult4 = F.interpolate(loss_mult4_expanded, size=Ll1.shape[1:], mode='nearest').squeeze(1)

        # loss_mult4= F.interpolate(loss_mult4, size=Ll1.shape[1:], mode='nearest').squeeze(1)


        loss = (1.0 - opt.lambda_dssim) * (Ll1 * loss_mult4).mean() +  opt.lambda_dssim * (ssim_loss * loss_mult4).mean() + uncertainty_loss
        
        loss = (1.0 - opt.lambda_dssim) * (Ll1 * loss_mult).mean() +  opt.lambda_dssim * (ssim_loss * loss_mult).mean() + uncertainty_loss

        Ll1 = Ll1.mean()
        loss.backward()
        
        iter_end.record()

        # 可视化所有张量
        if iteration % 100 == 0:  # 每100次迭代可视化一次
            
            # 确保所有张量在CPU上
            gt_cpu = gt_image.cpu() # gt
            prediction_cpu = image.cpu() # render
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
        



        # # jqq 保存 uncertainty loss图像
        # if iteration%10000==0:
        #     # iteration == 30001
        #     save_dir = scene.model_path +"/uncertainty_loss_image"
        #     os.makedirs(save_dir, exist_ok=True)
            
        #     all_viewpoint_stack = list(range(len(scene.getTrainCameras())))
        #     for i in tqdm(range(0,len(all_viewpoint_stack),1), dynamic_ncols=True, ascii=True):
        #         loss_viewpoint_cam = scene.getTrainCameras()[i]
        #         # image_path=self.train_image_paths[i]
        #         # image_id=int(Path(image_path).stem)
        #         # loss_gt_image = self.train_images[i].to(device)  # 已在正确设备上
        #         # loss_render_pkg = self.model._render_internal(
        #         #     loss_viewpoint_cam, 
        #         #     config=self.config, 
        #         #     embedding=embedding, 
        #         #     kernel_size=self.config.kernel_size
        #         # )
        #         # loss_image_toned: Tensor = loss_render_pkg["render"]
        #         # image_path = loss_viewpoint_cam.image_name
        #         loss_gt_image = loss_viewpoint_cam.original_image.cuda().unsqueeze(0) 
        #         voxel_visible_mask = prefilter_voxel(loss_viewpoint_cam, gaussians, pipe,background)
        #         retain_grad = False
        #         loss_render_pkg = render(
        #             loss_viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad
        #         )
                
        #         loss_image_toned = loss_render_pkg["render"].detach().unsqueeze(0)
                
                
        #         # 计算不确定性损失
        #         loss_uncertainty_loss, _, _, loss_loss_mult = gaussians.uncertainty_model._compute_losses(
        #             iteration, 
        #             loss_gt_image, 
        #             loss_image_toned, 
        #             prefix='', 
        #             _cache_entry=('train', loss_viewpoint_cam.colmap_id)
        #         )

        #         # 计算loss_mult的平均值
        #         loss_loss_mult_mean = loss_loss_mult.mean()
        #         # 将大于平均值的元素设置为True，其余为False
        #         loss_loss_mult = (loss_loss_mult > loss_loss_mult_mean).to(dtype=loss_loss_mult.dtype)

        #         #######################################################################
        #         # 处理不确定性图（转换到CPU进行可视化）
        #         uncertainty_map = loss_uncertainty_loss.detach().cpu().numpy()
                
        #         # 确保不确定性图为二维数组
        #         if uncertainty_map.ndim == 4:
        #             uncertainty_map = uncertainty_map[0, 0]
        #         elif uncertainty_map.ndim == 3:
        #             if uncertainty_map.shape[0] == 1:
        #                 uncertainty_map = uncertainty_map.squeeze(0)
        #             else:
        #                 uncertainty_map = uncertainty_map.mean(axis=0)
        #         else:
        #             raise ValueError(f"Unsupported shape: {uncertainty_map.shape}")
                
        #         # 创建热力图
        #         cmap = plt.get_cmap('jet')
        #         colored_map = cmap(uncertainty_map)
        #         rgb_map = (colored_map[:, :, :3] * 255).astype(np.uint8)
                
        #         #######################################################################
        #         # 处理loss_loss_mult：即mask
        #         loss_loss_mult = loss_loss_mult.squeeze(0)
        #         loss_mult_np = loss_loss_mult.detach().cpu().numpy()
        #         # 确保mask为二维数组并转换为图像
        #         if loss_mult_np.ndim == 3:
        #             loss_mult_np = loss_mult_np[0] if loss_mult_np.shape[0] == 1 else loss_mult_np.mean(axis=0)
                
        #         # 将mask转换为0-255的灰度图
        #         loss_mult_img = (loss_mult_np * 255).astype(np.uint8)
        #         loss_mult_img = Image.fromarray(loss_mult_img, mode='L').convert('RGB')

        #         #######################################################################
        #         # 处理loss_gt_image
        #         gt_np = loss_gt_image.cpu().numpy()
        #         if gt_np.ndim == 3:
        #             gt_np = np.transpose(gt_np, (1, 2, 0))
        #         elif gt_np.ndim == 4:
        #             gt_np = np.transpose(gt_np[0, ...], (1, 2, 0))
        #         if gt_np.dtype != np.uint8:
        #             gt_np = (gt_np * 255).astype(np.uint8)
        #         gt_img = Image.fromarray(gt_np)
                
        #         #######################################################################
        #         # 处理loss_image_toned
        #         render_np = loss_image_toned.detach().cpu().numpy()
        #         if render_np.ndim == 3:
        #             render_np = np.transpose(render_np, (1, 2, 0))
        #         elif render_np.ndim == 4:
        #             render_np = np.transpose(render_np[0, ...], (1, 2, 0))
        #         if render_np.dtype != np.uint8:
        #             render_np = (render_np * 255).astype(np.uint8)
        #         render_img = Image.fromarray(render_np)
                
        #         # 确保所有图像具有相同的高度
        #         height = max(rgb_map.shape[0], loss_mult_img.height, gt_img.height, render_img.height)
                
        #         # 调整图像大小以匹配高度
        #         uncertainty_img = Image.fromarray(rgb_map)
        #         uncertainty_img = uncertainty_img.resize((int(uncertainty_img.width * height / uncertainty_img.height), height))
        #         loss_mult_img = loss_mult_img.resize((int(loss_mult_img.width * height / loss_mult_img.height), height))
        #         gt_img = gt_img.resize((int(gt_img.width * height / gt_img.height), height))
        #         render_img = render_img.resize((int(render_img.width * height / render_img.height), height))

        #         # 分别存储四张图片
        #         # 假设已完成图像大小调整，得到以下四个Image对象：
        #         # uncertainty_img, loss_mult_img, gt_img, render_img

        #         # 定义存储路径（可根据需求修改）
        #         output_dir = f"{save_dir}/{iteration:05d}"
        #         os.makedirs(output_dir, exist_ok=True)  # 创建输出目录（若不存在）

        #         image_id = loss_viewpoint_cam.uid
        #         # 存储四张图片
        #         try:
        #             # 1. 存储不确定性图（Uncertainty Map）
        #             uncertainty_path = os.path.join(output_dir, f"{i:05d}_{image_id:04d}_uncertainty_map.png")
        #             uncertainty_img.save(uncertainty_path)
        #             # print(f"✅ 不确定性图已保存至：{uncertainty_path}")

        #             # 2. 存储损失权重图（Loss Multiplier Map）
        #             loss_mult_path = os.path.join(output_dir, f"{i:05d}_{image_id:04d}_mask.png")
        #             loss_mult_img.save(loss_mult_path)
        #             # print(f"✅ Mask图已保存至：{loss_mult_path}")

        #             # 3. 存储真实图像（Ground Truth Image）
        #             gt_path = os.path.join(output_dir,f"{i:05d}_{image_id:04d}_gt.png")
        #             gt_img.save(gt_path)
        #             # print(f"✅ 真实图像已保存至：{gt_path}")

        #             # 4. 存储渲染图像（Rendered Image）
        #             render_path = os.path.join(output_dir, f"{i:05d}_{image_id:04d}_render.png")
        #             render_img.save(render_path)
        #             # print(f"✅ 渲染图像已保存至：{render_path}")

        #         except Exception as e:
        #             print(f"❌ 图片存储失败：{str(e)}")


        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress", dynamic_ncols=True, ascii=True)):
        
        torch.cuda.synchronize();t_start = time.time()
        
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        config = Config()
        for key, value in dataset.__dict__.items():
            if hasattr(config, key):
                setattr(config, key, value)

        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, uncertainty_config=config)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

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
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    

    try:
        saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')
        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
