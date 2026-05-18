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

from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    extras,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    num_extra_channels,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        extras,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        num_extra_channels,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        extras,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        num_channels,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            extras,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                num_rendered, color, extra, depth, radii, geomBuffer, binningBuffer, imgBuffer = getattr(_C, f'rasterize_gaussians_{num_channels}')(*args)
            except Exception as ex:
                torch.save(cpu_args, 'snapshot_fw.dump')
                print('\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.')
                raise ex
        else:
            num_rendered, color, extra, depth, radii, geomBuffer, binningBuffer, imgBuffer = getattr(_C, f'rasterize_gaussians_{num_channels}')(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.num_channels = num_channels
        ctx.save_for_backward(extras, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, extra, radii, depth

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_extra, grad_radii, grad_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        num_channels = ctx.num_channels
        extras, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            raster_settings.bg,
            means3D,
            radii,
            extras,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            grad_out_extra,
            grad_depth,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                grad_means2D, grad_extra, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = getattr(_C, f'rasterize_gaussians_backward_{num_channels}')(*args)
            except Exception as ex:
                torch.save(cpu_args, 'snapshot_bw.dump')
                print('\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n')
                raise ex
        else:
            grad_means2D, grad_extra, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = getattr(_C, f'rasterize_gaussians_backward_{num_channels}')(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_extra,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None,
            None,
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(positions, raster_settings.viewmatrix, raster_settings.projmatrix)

        return visible

    def forward(self, means3D, means2D, opacities, shs=None, extras=None, scales=None, rotations=None, cov3D_precomp=None):

        if extras is None:
            num_channels = 0
        else:
            num_channels = extras.shape[-1]

        raster_settings = self.raster_settings

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([]).cuda()
        if extras is None:
            extras = torch.Tensor([]).cuda()

        if scales is None:
            scales = torch.Tensor([]).cuda()
        if rotations is None:
            rotations = torch.Tensor([]).cuda()
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([]).cuda()

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            extras,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
            num_channels,
        )
