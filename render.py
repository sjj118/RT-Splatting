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

import ast
import os
from argparse import ArgumentParser
from functools import partial

import open3d as o3d
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.easyvolcap_utils import load_traj_from_easyvolcap, save_traj_to_easyvolcap
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from utils.render_utils import create_videos, generate_path


def parse_render_path_kwargs(raw_kwargs: str):
    try:
        parsed_kwargs = ast.literal_eval(raw_kwargs)
    except (ValueError, SyntaxError) as exc:
        raise ValueError("--render_path_kwargs must be a Python dict literal, e.g. \"{'n_rots': 3, 'z_variation': 0.5}\".") from exc

    if parsed_kwargs is None:
        return {}
    if not isinstance(parsed_kwargs, dict):
        raise ValueError('--render_path_kwargs must evaluate to a dict.')
    return parsed_kwargs


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description='Testing script parameters')
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--iteration', default=-1, type=int)
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--skip_mesh', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--render_path', action='store_true')
    parser.add_argument('--load_path', default='', type=str, help='Render path: load EasyVolcap intri.yml/extri.yml from this directory instead of generating path')
    parser.add_argument('--zoom', default=1.0, type=float, help='Render path: zoom factor for path generation')
    parser.add_argument('--path_frames', default=480, type=int, help='Render path: number of frames')
    parser.add_argument('--path_type', default='ellipse', choices=['ellipse', 'spiral'], help='Render path: choose generate_ellipse_path or generate_spiral_path in utils/render_utils.py')
    parser.add_argument('--path_kwargs', default='{}', type=str, help='Render path: Python dict literal for extra kwargs passed to path generation function')
    parser.add_argument('--voxel_size', default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument('--depth_trunc', default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument('--sdf_trunc', default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument('--num_cluster', default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument('--unbounded', action='store_true', help='Mesh: using unbounded mode for meshing')
    parser.add_argument('--mesh_res', default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    args = get_combined_args(parser)
    print('Rendering ' + args.model_path)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, dataset)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    train_dir = os.path.join(args.model_path, 'train', 'ours_{}'.format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', 'ours_{}'.format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, partial(render, pipe=pipe), bg_color=background)

    if not args.skip_train:
        print('export training images ...')
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTrainCameras(), train_dir)

    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print('export rendered testing images ...')
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras(), test_dir)

    if args.render_path:
        print('render videos ...')
        traj_dir = os.path.join(args.model_path, 'traj', 'ours_{}'.format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        if args.load_path:
            print(f'loading camera trajectory from {args.load_path} ...')
            cam_traj = load_traj_from_easyvolcap(args.load_path, scene.getTrainCameras())
        else:
            path_kwargs = parse_render_path_kwargs(args.path_kwargs)
            cam_traj = generate_path(
                scene.getTrainCameras(),
                n_frames=args.path_frames,
                zoom_factor=args.zoom,
                path_type=args.path_type,
                path_kwargs=path_kwargs,
            )
        save_traj_to_easyvolcap(cam_traj, traj_dir)
        gaussExtractor.reconstruction(cam_traj, traj_dir)
        create_videos(base_dir=traj_dir, input_dir=traj_dir, out_name='render_traj', num_frames=len(cam_traj))

    if not args.skip_mesh:
        print('export mesh ...')
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(scene.getTrainCameras())
        gaussExtractor.estimate_bounding_sphere()
        # extract the mesh and save
        if args.unbounded:
            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = 'fuse.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0 else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print('mesh saved at {}'.format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print('mesh post processed saved at {}'.format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))
