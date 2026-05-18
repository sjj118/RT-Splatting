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

import matplotlib
import torch
import torch.nn.functional as F


def mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def apply_colormap(image, cmap='turbo'):
    colormap = matplotlib.colormaps[cmap]
    return torch.tensor(colormap(image.squeeze(0).cpu().numpy())[..., :3]).permute(2, 0, 1)


def linear_normalize(image, min=None, max=None):
    mask = torch.logical_and(~image.isnan(), ~image.isinf())
    if min is None:
        min = image[mask].quantile(0.01)
    if max is None:
        max = image[mask].quantile(0.99)
    image = (image - min) / (max - min + 1e-10)
    image[~mask] = 0
    return image.clip(0, 1)


def log_normalize(image):
    image = torch.log(image + 1e-6)
    return linear_normalize(image)


def gradient_map(image):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda() / 4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda() / 4

    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    magnitude = magnitude.norm(dim=0, keepdim=True)

    return magnitude


def sobel_operator(image, norm=2):
    """Compute Sobel gradients for the input image tensor.

    Args:
        image: Tensor of shape [C, H, W] or [B, C, H, W]

    Returns:
        Tensor of gradients with same shape as input
    """
    is_batch = True
    if image.dim() == 3:
        is_batch = False
        image = image.unsqueeze(0)

    # Sobel kernels
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device) / 4
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=image.device) / 4

    kernel_x = kernel_x.view(1, 1, 3, 3).repeat(image.size(1), 1, 1, 1)
    kernel_y = kernel_y.view(1, 1, 3, 3).repeat(image.size(1), 1, 1, 1)

    pad = F.pad(image, (1, 1, 1, 1), mode='reflect')
    grad_x = F.conv2d(pad, kernel_x, groups=image.size(1))
    grad_y = F.conv2d(pad, kernel_y, groups=image.size(1))

    if norm:
        grad = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)
        grad = grad.norm(dim=1, p=norm, keepdim=True)
    else:
        grad = torch.stack([grad_x, grad_y], dim=1)

    return grad if is_batch else grad.squeeze(0)


def laplacian_operator(image, norm=2):
    """
    Compute the Laplacian gradient of the input image.

    Args:
        image: Tensor, shape [C, H, W] or [B, C, H, W]

    Returns:
        Laplacian gradient, with the same shape as the input
    """
    is_batch = True
    if image.dim() == 3:
        is_batch = False
        image = image.unsqueeze(0)

    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=image.device) / 4
    kernel = kernel.view(1, 1, 3, 3).repeat(image.size(1), 1, 1, 1)
    pad = F.pad(image, (1, 1, 1, 1), mode='reflect')
    lap = F.conv2d(pad, kernel, groups=image.size(1))
    if norm:
        lap = lap.norm(dim=1, p=norm, keepdim=True)
    return lap if is_batch else lap.squeeze(0)


def delta_operator(image, norm=None):
    """
    Compute the discrete gradient of the input image.

    Args:
        image: Tensor of shape [C, H, W] or [B, C, H, W]
        norm: Whether to return the norm of the gradient.

    Returns:
        (grad_x, grad_y): gradient of image along x (height) and y (width)
    """
    is_batch = True
    if image.dim() == 3:
        is_batch = False
        image = image.unsqueeze(0)

    grad_x = image[:, :, 1:, :] - image[:, :, :-1, :]
    grad_y = image[:, :, :, 1:] - image[:, :, :, :-1]

    grad = torch.stack(
        [
            F.pad(grad_x, (0, 0, 1, 0)),
            F.pad(grad_x, (0, 0, 0, 1)),
            F.pad(grad_y, (1, 0, 0, 0)),
            F.pad(grad_y, (0, 1, 0, 0)),
        ],
        dim=1,
    )

    if norm:
        grad = grad.norm(dim=1).norm(dim=1, p=norm, keepdim=True)

    return grad if is_batch else grad.squeeze(0)


def local_variance(image, weights=None, kernel_size=3):
    """Compute the per-pixel variance over a square window.

    Args:
        image: Tensor shaped [C, H, W] or [B, C, H, W].
        weights: Optional tensor shaped [1, H, W] or [B, 1, H, W] used as per-pixel weights.
        kernel_size: Size of the (square) averaging window.
        unbiased: If True, applies Bessel's correction.

    Returns:
        Tensor of local variance with the same shape as the input.
    """

    is_batch = True
    if image.dim() == 3:
        is_batch = False
        image = image.unsqueeze(0)

    if weights is None:
        weights = torch.ones_like(image)[:, 0:1, ...]
    elif weights.dim() == 3:
        weights = weights.unsqueeze(0)
    weights = weights.to(device=image.device, dtype=image.dtype)

    kernel_size = int(kernel_size)
    pad = max(kernel_size // 2, 0)

    if pad > 0:
        padded_img = F.pad(image, (pad, pad, pad, pad), mode='reflect')
        padded_w = F.pad(weights, (pad, pad, pad, pad), mode='reflect')
    else:
        padded_img = image
        padded_w = weights

    channels = image.size(1)
    eps = 1e-12
    kernel = torch.ones((channels, 1, kernel_size, kernel_size), device=image.device, dtype=image.dtype)

    weight_sum = F.conv2d(padded_w, kernel, padding=0, groups=1).clamp_min(eps)
    weighted_sum = F.conv2d(padded_img * padded_w, kernel, padding=0, groups=channels)
    weighted_sq_sum = F.conv2d(padded_img.pow(2) * padded_w, kernel, padding=0, groups=channels)

    variance = ((weighted_sq_sum - weighted_sum.pow(2) / weight_sum) / (kernel_size**2)).clamp_min(0.0)
    std = variance.pow(0.5)

    return std if is_batch else std.squeeze(0)
