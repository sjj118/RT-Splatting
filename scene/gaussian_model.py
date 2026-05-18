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

import numpy as np
import nvdiffrast.torch
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from utils.general_utils import build_rotation, build_scaling_rotation, get_expon_lr_func, inverse_sigmoid
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from utils.sph_utils import *
from utils.system_utils import mkdir_p


class SphMipEncoding(nn.Module):
    def __init__(self, n_levels: int = 8, plane_size: int = 512, feature_dim: int = 16, Sn: int = 1, dim: int = 1, rand_init: bool = False):
        super(SphMipEncoding, self).__init__()
        self.n_levels = n_levels
        self.plane_size = plane_size

        self.register_parameter(
            'fm',
            nn.Parameter(torch.zeros(Sn, dim, plane_size, 2 * plane_size, feature_dim)),
        )

        if rand_init:
            self.init_parameters()

    def init_parameters(self) -> None:
        nn.init.uniform_(self.fm, -1e-2, 1e-2)

    def forward(self, x, level, index=0, weight=False):
        """
        x: [0,1], Nx3
        level: [0, max_level], Nx1
        """
        x[..., 0] = x[..., 0] * 0.5 + 0.25

        decomposed_x = x

        level = torch.broadcast_to(level, decomposed_x.shape[:3]).contiguous()

        fm = self.fm[index]  # [N, L, 2L, feat_dim]

        padding_fm = torch.cat([fm[:, :, self.plane_size :, :], fm, fm[:, :, : self.plane_size, :]], dim=2)
        enc = nvdiffrast.torch.texture(
            padding_fm,
            decomposed_x,
            mip_level_bias=level * self.n_levels,
            boundary_mode='clamp',
            max_mip_level=self.n_levels - 1,
        )

        enc = (
            enc.permute(1, 2, 0, 3)
            .contiguous()
            .view(
                x.shape[0],
                -1,
            )
        )
        return enc


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0, 2, 1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device='cuda')
            trans[:, :3, :3] = RS
            trans[:, 3, :3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.occupancy_activation = torch.sigmoid
        self.inverse_occupancy_activation = inverse_sigmoid
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.transmissivity_activation = torch.sigmoid
        self.inverse_transmissivity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._occupancy = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.last_update = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self._roughness = torch.empty(0)
        self._opacity = torch.empty(0)
        self._transmissivity = torch.empty(0)

        if args.env_scope_radius > 0:
            self.ENV_CENTER = torch.tensor([float(c) for c in args.env_scope_center], device='cuda')
            self.ENV_RADIUS = args.env_scope_radius
            self.XYZ = [int(float(c)) for c in args.xyz_axis]

            print(self.ENV_CENTER, self.ENV_RADIUS, self.XYZ)
        else:
            self.ENV_CENTER = torch.tensor([0, 0, 0], device='cuda')
            self.ENV_RADIUS = torch.inf
            self.XYZ = [0, 1, 2]

        #################################################################################################
        n_levels = 9
        plane_size = 2 ** (n_levels)

        self.sph_dim = 16
        self.dim = 1

        self.dir_encoding = SphMipEncoding(n_levels, plane_size, self.sph_dim, 1, self.dim, args.rand_init).cuda()
        print('SphMipEncoding ### level:', n_levels, '  size:', plane_size)

        self.gsfeat_dim = 4
        run_dim = args.run_dim

        self.light_mlp = nn.Sequential(
            nn.Linear(self.sph_dim * self.gsfeat_dim + self.sph_dim, run_dim),
            nn.ReLU(inplace=True),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(inplace=True),
            nn.Linear(run_dim, 4),
        ).cuda()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._occupancy,
            self._transmissivity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.last_update,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._occupancy,
            self._transmissivity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            last_update,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.last_update = last_update
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_reflectance(self):
        return torch.sigmoid(self._reflectance + inverse_sigmoid(0.5))

    @property
    def get_inside_mask(self):
        return torch.sum((self.get_xyz - self.ENV_CENTER) ** 2, dim=-1, keepdim=True) < self.ENV_RADIUS**2

    @property
    def get_roughness(self):
        return torch.sigmoid(self._roughness)

    @property
    def get_language_feature(self):
        return torch.tanh(self._language_feature)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_occupancy(self):
        return self.occupancy_activation(self._occupancy)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_transmissivity(self):
        return self.transmissivity_activation(self._transmissivity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print('Number of points at initialisation : ', fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001) * 0.1
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device='cuda')

        occupancies = self.inverse_occupancy_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device='cuda'))
        opacities = self.inverse_opacity_activation(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device='cuda'))
        transmissivities = self.inverse_transmissivity_activation(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device='cuda'))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._occupancy = nn.Parameter(occupancies.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._transmissivity = nn.Parameter(transmissivities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device='cuda')

        self._roughness = nn.Parameter((torch.zeros((fused_point_cloud.shape[0], 1), device='cuda')).requires_grad_(True))
        self._reflectance = nn.Parameter((torch.zeros((fused_point_cloud.shape[0], 1), device='cuda')).requires_grad_(True))
        self._language_feature = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], self.gsfeat_dim), device='cuda').requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.last_update = torch.zeros((self.get_xyz.shape[0], 1), device='cuda', dtype=torch.int)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, 'name': 'xyz'},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, 'name': 'f_dc'},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, 'name': 'f_rest'},
            {'params': [self._occupancy], 'lr': training_args.occupancy_lr, 'name': 'occupancy'},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, 'name': 'opacity'},
            {'params': [self._transmissivity], 'lr': training_args.transmissivity_lr, 'name': 'transmissivity'},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, 'name': 'scaling'},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, 'name': 'rotation'},
        ]

        l.extend(
            [
                {'params': [self._roughness], 'lr': training_args.roughness_lr, 'name': 'roughness'},
                {'params': [self._reflectance], 'lr': training_args.reflectance_lr, 'name': 'reflectance'},
                {'params': [self._language_feature], 'lr': training_args.feature_lr, 'name': 'feature'},
                {'params': list(self.light_mlp.parameters()), 'lr': training_args.mlp_lr, 'name': 'light_mlp'},
                {'params': list(self.dir_encoding.parameters()), 'lr': training_args.encoding_lr, 'name': 'dir_encoding'},
            ]
        )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'xyz':
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def update_learning_rate_by_name(self, name: str, lr: float):
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == name:
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = [
            'x',
            'y',
            'z',
        ]

        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))

        l.append('occupancy')
        l.append('opacity')
        l.append('transmissivity')

        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        for i in range(self._roughness.shape[1]):
            l.append('roughness_{}'.format(i))
        for i in range(self._reflectance.shape[1]):
            l.append('reflectance_{}'.format(i))

        for i in range(self._language_feature.shape[1]):
            l.append('feature_{}'.format(i))

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()

        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        occupancies = self._occupancy.detach().cpu().numpy()
        opacity = self._opacity.detach().cpu().numpy()
        transmissivity = self._transmissivity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        roughness = self._roughness.detach().cpu().numpy()
        reflectance = self._reflectance.detach().cpu().numpy()

        language_feature = self._language_feature.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, f_rest, occupancies, opacity, transmissivity, scale, rotation, roughness, reflectance, language_feature), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        torch.save(self.light_mlp, path.split('point_cloud.ply')[0] + '/light_mlp.pt')
        torch.save(self.dir_encoding, path.split('point_cloud.ply')[0] + '/dir_encoding.pt')

    def reset_occupancy(self):
        self._occupancy.data[torch.isnan(self._occupancy.data.mean(dim=-1))] = 0.0
        self._occupancy.data[torch.isnan(self._xyz.data.mean(dim=-1))] = 0.0
        self._occupancy.data[torch.isnan(self._scaling.data.mean(dim=-1))] = 0.0
        self._occupancy.data[torch.isnan(self._rotation.data.mean(dim=-1))] = 0.0

        occupancies_new = self.inverse_occupancy_activation(torch.min(self.get_occupancy, torch.ones_like(self.get_occupancy) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(occupancies_new, 'occupancy')
        self._occupancy = optimizable_tensors['occupancy']

    def reset_opacity(self):
        self._opacity.data[torch.isnan(self._opacity.data.mean(dim=-1))] = 0.0
        self._opacity.data[torch.isnan(self._xyz.data.mean(dim=-1))] = 0.0
        self._opacity.data[torch.isnan(self._scaling.data.mean(dim=-1))] = 0.0
        self._opacity.data[torch.isnan(self._rotation.data.mean(dim=-1))] = 0.0

        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, 'opacity')
        self._opacity = optimizable_tensors['opacity']

    def load_ply(self, path):

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]['x']), np.asarray(plydata.elements[0]['y']), np.asarray(plydata.elements[0]['z'])), axis=1)

        occupancies = np.asarray(plydata.elements[0]['occupancy'])[..., np.newaxis]
        opacities = np.asarray(plydata.elements[0]['opacity'])[..., np.newaxis]
        transmissivities = np.asarray(plydata.elements[0]['transmissivity'])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]['f_dc_0'])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]['f_dc_1'])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]['f_dc_2'])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('f_rest_')]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('scale_')]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('rot')]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        roughness_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('roughness')]
        roughness = np.zeros((xyz.shape[0], len(roughness_names)))
        for idx, attr_name in enumerate(roughness_names):
            roughness[:, idx] = np.asarray(plydata.elements[0][attr_name])

        reflectance_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('reflectance')]
        reflectance = np.zeros((xyz.shape[0], len(reflectance_names)))
        for idx, attr_name in enumerate(reflectance_names):
            reflectance[:, idx] = np.asarray(plydata.elements[0][attr_name])

        language_feature_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('feature')]
        language_feature_names = sorted(language_feature_names, key=lambda x: int(x.split('_')[-1]))
        language_feature = np.zeros((xyz.shape[0], len(language_feature_names)))
        for idx, attr_name in enumerate(language_feature_names):
            language_feature[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device='cuda').requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device='cuda').transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device='cuda').transpose(1, 2).contiguous().requires_grad_(True))
        self._occupancy = nn.Parameter(torch.tensor(occupancies, dtype=torch.float, device='cuda').requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device='cuda').requires_grad_(True))
        self._transmissivity = nn.Parameter(torch.tensor(transmissivities, dtype=torch.float, device='cuda').requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device='cuda').requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device='cuda').requires_grad_(True))

        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device='cuda').requires_grad_(True))
        self._reflectance = nn.Parameter(torch.tensor(reflectance, dtype=torch.float, device='cuda').requires_grad_(True))

        self._language_feature = nn.Parameter(torch.tensor(language_feature, dtype=torch.float, device='cuda').requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device='cuda')

        self.light_mlp = torch.load(path.split('point_cloud.ply')[0] + '/light_mlp.pt')
        self.dir_encoding = torch.load(path.split('point_cloud.ply')[0] + '/dir_encoding.pt')
        print('Load Path', path)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] == 'light_mlp' or group['name'] == 'dir_encoding':
                continue

            if group['name'] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state['exp_avg'] = torch.zeros_like(tensor)
                stored_state['exp_avg_sq'] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] == 'light_mlp' or group['name'] == 'dir_encoding':
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = stored_state['exp_avg'][mask]
                stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter((group['params'][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group['name']] = group['params'][0]
            else:
                group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors['xyz']
        self._features_dc = optimizable_tensors['f_dc']
        self._features_rest = optimizable_tensors['f_rest']
        self._occupancy = optimizable_tensors['occupancy']
        self._opacity = optimizable_tensors['opacity']
        self._transmissivity = optimizable_tensors['transmissivity']
        self._scaling = optimizable_tensors['scaling']
        self._rotation = optimizable_tensors['rotation']

        self._reflectance = optimizable_tensors['reflectance']
        self._roughness = optimizable_tensors['roughness']

        self._language_feature = optimizable_tensors['feature']

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.last_update = self.last_update[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] == 'light_mlp' or group['name'] == 'dir_encoding':
                continue

            assert len(group['params']) == 1
            extension_tensor = tensors_dict[group['name']]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = torch.cat((stored_state['exp_avg'], torch.zeros_like(extension_tensor)), dim=0)
                stored_state['exp_avg_sq'] = torch.cat((stored_state['exp_avg_sq'], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group['name']] = group['params'][0]
            else:
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group['name']] = group['params'][0]

        return optimizable_tensors

    def densification_postfix(
        self, new_xyz, new_features_dc, new_features_rest, new_occupancies, new_opacity, new_transmissivity, new_scaling, new_rotation, new_reflectance, new_roughness, new_language_feature
    ):
        d = {
            'xyz': new_xyz,
            'f_dc': new_features_dc,
            'f_rest': new_features_rest,
            'occupancy': new_occupancies,
            'opacity': new_opacity,
            'transmissivity': new_transmissivity,
            'scaling': new_scaling,
            'rotation': new_rotation,
            'reflectance': new_reflectance,
            'roughness': new_roughness,
            'feature': new_language_feature,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors['xyz']
        self._features_dc = optimizable_tensors['f_dc']
        self._features_rest = optimizable_tensors['f_rest']
        self._occupancy = optimizable_tensors['occupancy']
        self._opacity = optimizable_tensors['opacity']
        self._transmissivity = optimizable_tensors['transmissivity']
        self._scaling = optimizable_tensors['scaling']
        self._rotation = optimizable_tensors['rotation']

        self._reflectance = optimizable_tensors['reflectance']
        self._roughness = optimizable_tensors['roughness']

        self._language_feature = optimizable_tensors['feature']

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device='cuda')

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device='cuda')
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_occupancy = self._occupancy[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_transmissivity = self._transmissivity[selected_pts_mask].repeat(N, 1)

        new_reflectance = self._reflectance[selected_pts_mask].repeat(N, 1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)

        new_language_feature = self._language_feature[selected_pts_mask].repeat(N, 1)

        self.last_update = torch.cat((self.last_update, self.last_update[selected_pts_mask].repeat(N, 1)), dim=0)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_occupancy, new_opacity, new_transmissivity, new_scaling, new_rotation, new_reflectance, new_roughness, new_language_feature)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device='cuda', dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_occupancies = self._occupancy[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_transmissivity = self._transmissivity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_reflectance = self._reflectance[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask]

        new_language_feature = self._language_feature[selected_pts_mask]

        self.last_update = torch.cat((self.last_update, self.last_update[selected_pts_mask]), dim=0)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_occupancies, new_opacity, new_transmissivity, new_scaling, new_rotation, new_reflectance, new_roughness, new_language_feature)

    def densify_and_prune(self, max_grad, min_occupancy, extent, max_screen_size, last_reset_iter):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_occupancy < min_occupancy).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        prune_mask = torch.logical_and(prune_mask, self.last_update.squeeze() > last_reset_iter)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, iteration):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        self.last_update[update_filter] = iteration
