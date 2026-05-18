import math

import numpy as np
import torch
import torch.nn.functional as F
from diff_surfel_anych import GaussianRasterizationSettings, GaussianRasterizer

from scene.gaussian_model import GaussianModel
from utils.camera_utils import *
from utils.color_utils import *
from utils.general_utils import *
from utils.point_utils import *
from utils.sph_utils import *


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device='cuda') + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    occupancy = pc.get_occupancy
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features

    # Forward pass

    extras = torch.cat([opacity], dim=-1)

    render_tran, volume_extras, _, volume_allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        extras=extras,
        opacities=occupancy * opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    (volume_opacity,) = volume_extras.split([1], dim=0)

    volume_alpha = volume_allmap[1:2]

    volume_normal = volume_allmap[2:5]
    volume_normal = (volume_normal.movedim(0, -1) @ (viewpoint_camera.world_view_transform[:3, :3].T)).movedim(-1, 0)
    volume_normal = F.normalize(volume_normal, dim=0)

    volume_depth_median = volume_allmap[5:6]
    volume_depth_median = torch.nan_to_num(volume_depth_median, 0, 0)

    volume_depth_expected = volume_allmap[0:1]
    volume_depth_expected = volume_depth_expected / volume_alpha
    volume_depth_expected = torch.nan_to_num(volume_depth_expected, 0, 0)

    volume_depth = volume_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * volume_depth_median

    volume_depth_normal = depth_to_normal_sobel(viewpoint_camera, volume_depth.movedim(0, -1)).movedim(-1, 0)
    volume_depth_normal = volume_depth_normal * volume_alpha.detach()

    volume_dist = volume_allmap[6:7]

    # Deferred pass

    extras = torch.cat([pc.get_roughness, pc.get_language_feature, pc.get_inside_mask, pc.get_inside_mask * pc.get_reflectance, opacity, pc.get_inside_mask * pc.get_transmissivity], dim=-1)

    render_scat, surface_extras, radii, surface_allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        extras=extras,
        opacities=occupancy,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    render_roughness, render_feature, foreground, render_reflectance, surface_opacity, render_transmissivity = surface_extras.split([1, 4, 1, 1, 1, 1], dim=0)
    foreground = foreground.detach()

    surface_alpha = surface_allmap[1:2]

    surface_normal = surface_allmap[2:5]
    surface_normal = (surface_normal.movedim(0, -1) @ (viewpoint_camera.world_view_transform[:3, :3].T)).movedim(-1, 0)
    surface_normal = F.normalize(surface_normal, dim=0)

    surface_depth_median = surface_allmap[5:6]
    surface_depth_median = torch.nan_to_num(surface_depth_median, 0, 0)

    surface_depth_expected = surface_allmap[0:1]
    surface_depth_expected = surface_depth_expected / surface_alpha
    surface_depth_expected = torch.nan_to_num(surface_depth_expected, 0, 0)

    surface_depth = surface_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * surface_depth_median

    surface_depth_normal = depth_to_normal_sobel(viewpoint_camera, surface_depth.movedim(0, -1)).movedim(-1, 0)
    surface_depth_normal = surface_depth_normal * surface_alpha.detach()

    surface_dist = surface_allmap[6:7]

    #####################################################################################################################

    with torch.no_grad():
        select_index = (foreground.flatten() > 0.05).nonzero(as_tuple=True)[0]

    render_spec = torch.zeros(3, image_height, image_width).cuda()
    render_attenuation = torch.zeros(1, image_height, image_width).cuda()

    _, viewdirs = camera_rays(viewpoint_camera)
    viewdirs = F.normalize(viewdirs, dim=-1)
    normal_map = surface_normal.movedim(0, -1)
    wo = F.normalize(reflect(-viewdirs, normal_map), dim=-1)

    if len(select_index) > 0:
        wo = wo.reshape(-1, 3)[select_index]
        normal_map = normal_map.reshape(-1, 3)[select_index]
        roughness_map = render_roughness.movedim(0, -1).reshape(-1, 1)[select_index]

        feature_map = render_feature.movedim(0, -1).reshape(-1, pc.gsfeat_dim)[select_index]
        feature_map = F.normalize(feature_map, dim=-1)

        feature_map = feature_map.reshape(-1, 1, pc.gsfeat_dim)
        feature_dirc = feature_map.reshape(-1, pc.gsfeat_dim)

        """ Sph-Mip """
        wo_xy = (cart2sph(wo.reshape(-1, 3)[..., pc.XYZ])[..., 1:] / torch.Tensor([[np.pi, 2 * np.pi]]).cuda())[..., [1, 0]]
        wo_xyz = torch.stack(
            [wo_xy[:, None, :]],
            dim=0,
        )

        spec_level = roughness_map.reshape(-1, 1)

        spec_feat = pc.dir_encoding(wo_xyz, spec_level.view(-1, 1), index=0).reshape(-1, pc.sph_dim)
        spec_feat_wrap = spec_feat.reshape(-1, pc.sph_dim, 1)
        spec_feat_dirc = spec_feat.reshape(-1, pc.sph_dim)

        # Specular color
        wrap_input = (spec_feat_wrap @ feature_map).reshape(-1, pc.sph_dim * pc.gsfeat_dim)
        input_mlp = torch.cat([wrap_input, spec_feat_dirc], -1)
        mlp_output = pc.light_mlp(input_mlp)
        spec_light = torch.exp(mlp_output[..., :3] + np.log(0.5))
        spec_attenuation = torch.sigmoid(mlp_output[..., 3:4])

        render_spec.reshape(3, -1)[:, select_index] = spec_light.transpose(0, 1)
        render_attenuation.reshape(1, -1)[:, select_index] = spec_attenuation.transpose(0, 1)

    render_attenuation = 1 - (1 - render_attenuation) * foreground

    final_tran = render_tran * render_transmissivity
    final_scat = render_scat * (1 - render_transmissivity)
    final_spec = render_spec * render_reflectance
    final_rendering = final_tran + final_scat

    if not pipe.init_stage:
        final_tran = final_tran * render_attenuation
        final_scat = final_scat * render_attenuation
        final_rendering = final_tran + final_scat + final_spec

    rets = {
        'final_rendering': final_rendering,
        'final_tran': final_tran,
        'final_scat': final_scat,
        'final_spec': final_spec,
        'render_spec': render_spec,
        'render_tran': render_tran,
        'render_scat': render_scat,
        'feature': render_feature,
        'roughness': render_roughness,
        'reflectance': render_reflectance,
        'transmissivity': render_transmissivity,
        'attenuation': render_attenuation,
        'foreground': foreground,
        'surface_alpha': surface_alpha,
        'surface_depth': surface_depth,
        'surface_normal': surface_normal,
        'surface_depth_normal': surface_depth_normal,
        'surface_dist': surface_dist,
        'surface_opacity': surface_opacity,
        'volume_alpha': volume_alpha,
        'volume_depth': volume_depth,
        'volume_normal': volume_normal,
        'volume_depth_normal': volume_depth_normal,
        'volume_dist': volume_dist,
        'volume_opacity': volume_opacity,
        'viewspace_points': means2D,
        'visibility_filter': radii > 0,
        'radii': radii,
    }

    return rets
