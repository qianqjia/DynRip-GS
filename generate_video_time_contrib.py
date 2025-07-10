import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer.render_time_contrib import render, prefilter_voxel
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model_uw_contrib import GaussianModel
from scene.cameras import Camera
import cv2
import torch.nn.functional as F
import numpy as np
import imageio

def rotation_matrix_to_quaternion(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = np.array([
            0.25 / s,
            (R[2, 1] - R[1, 2]) * s,
            (R[0, 2] - R[2, 0]) * s,
            (R[1, 0] - R[0, 1]) * s
        ])
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q = np.array([
                (R[2, 1] - R[1, 2]) / s,
                0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s
            ])
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q = np.array([
                (R[0, 2] - R[2, 0]) / s,
                (R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s
            ])
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q = np.array([
                (R[1, 0] - R[0, 1]) / s,
                (R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s,
                0.25 * s
            ])
    return q

def slerp(q1, q2, t):
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        result = result / np.linalg.norm(result)
        return result

    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    q2 = q2 - dot * q1
    q2 = q2 / np.linalg.norm(q2)
    q = q1 * np.cos(theta) + q2 * np.sin(theta)
    return q

def quaternion_to_rotation_matrix(q):
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)]
    ])
    return R


def tensor2image(tensor):
    return tensor.detach().permute(1, 2, 0).cpu().numpy()

def record_set(views, gaussians, pipeline, background, out_path='out', is_train=True, is_move=True, is_static=False, light_mode=None):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30
    frame_size = (views[0].image_width, views[0].image_height)

    move_str = 'MovePos' if is_move else 'StaticPos'
    time_str = 'FixedTime' if is_static else 'ChangeTime'
    obj = None
    if light_mode is None:
        obj = 'both'
    elif light_mode == 'static':
        obj = 'obj'
    else:
        obj = 'light'

    out = cv2.VideoWriter(out_path + '_{}_{}_{}_{}.mp4'.format('train' if is_train else 'test', move_str, time_str, obj), fourcc, fps, frame_size)

    R0 = views[0].R
    T0 = views[0].T
    for view_pre, view_cur in zip(views[:-1], views[1:]):
        R_pre = view_pre.R
        R_cur = view_cur.R
        t_pre = view_pre.T
        t_cur = view_cur.T
        uid_pre = view_pre.uid
        uid_cur = view_cur.uid
        colmap_id_pre = view_pre.colmap_id
        colmap_id_cur = view_cur.colmap_id
        q_pre = rotation_matrix_to_quaternion(R_pre)
        q_cur = rotation_matrix_to_quaternion(R_cur)
        for t in range(20):
            t = t / 20.0
            q_inter = slerp(q_pre, q_cur, t)
            R_inter = quaternion_to_rotation_matrix(q_inter)
            T_inter = (1 - t) * t_pre + t * t_cur
            uid = (1 - t) * uid_pre + t * uid_cur
            
            if is_static:
                colmap_id = 0
            else:
                colmap_id = (1 - t) * colmap_id_pre + t * colmap_id_cur
            
            if is_move:
                view = Camera(colmap_id, R_inter, T_inter, view_cur.FoVx, view_cur.FoVy, view_pre.original_image, None, None, uid=uid)
            else:
                view = Camera(colmap_id, R0, T0, view_cur.FoVx, view_cur.FoVy, view_pre.original_image, None, None, uid=uid)

            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
            render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, mode=light_mode)
            rendering = render_pkg["render"]
            color = cv2.cvtColor((np.clip(tensor2image(rendering), 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
            out.write(color)
            
    out.release()


def render_record(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, out_path = 'out'):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not skip_train:
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=True, is_move=True, is_static=False, light_mode=None)
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=True, is_move=True, is_static=False, light_mode='static')
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=True, is_move=True, is_static=False, light_mode='light')

        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=True, is_move=True, is_static=True, light_mode=None)
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=True, is_move=True, is_static=True, light_mode='static')
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=True, is_move=True, is_static=True, light_mode='light')

        
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=True, is_move=False, is_static=False, light_mode=None)
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=True, is_move=False, is_static=False, light_mode='static')
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=True, is_move=False, is_static=False, light_mode='light')

        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=True, is_move=False, is_static=True, light_mode=None)
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=True, is_move=False, is_static=True, light_mode='static')
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=True, is_move=False, is_static=True, light_mode='light')
    
    if not skip_test:
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=False, is_move=True, is_static=False, light_mode=None)
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=False, is_move=True, is_static=False, light_mode='static')
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=False, is_move=True, is_static=False, light_mode='light')

        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=False, is_move=True, is_static=True, light_mode=None)
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=False, is_move=True, is_static=True, light_mode='static')
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=False, is_move=True, is_static=True, light_mode='light')

        
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=False, is_move=False, is_static=False, light_mode=None)
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=False, is_move=False, is_static=False, light_mode='static')
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=False, is_move=False, is_static=False, light_mode='light')

        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=False, is_move=False, is_static=True, light_mode=None)
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=False, is_move=False, is_static=True, light_mode='static')
        record_set(scene.getTrainCameras(), gaussians, pipeline, background, out_path, is_train=False, is_move=False, is_static=True, light_mode='light')


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--out_path", default='out.mp4', type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_record(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.out_path)