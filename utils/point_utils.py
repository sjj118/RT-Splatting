import torch
import torch.nn.functional as F


def camera_rays(view):
    """
    Returns:
        rays_o: (3,)
        rays_d: (H, W, 3)
    """
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([[W / 2, 0, 0, (W) / 2], [0, H / 2, 0, (H) / 2], [0, 0, 0, 1]]).float().cuda().T
    intrins = (view.projection_matrix @ ndc2pix)[:3, :3].T

    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1)
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    return rays_o, rays_d


def depths_to_points(view, depthmap):
    """
    Args:
        depthmap: (B, H, W, 1)
    Returns:
        points: (B, H, W, 3)
    """
    rays_o, rays_d = camera_rays(view)
    points = depthmap * rays_d + rays_o
    return points


def depth_to_normal(view, depth):
    points = depths_to_points(view, depth)
    output = torch.zeros_like(points)
    dx = torch.cat([points[..., 2:, 1:-1, :] - points[..., :-2, 1:-1, :]], dim=0)
    dy = torch.cat([points[..., 1:-1, 2:, :] - points[..., 1:-1, :-2, :]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[..., 1:-1, 1:-1, :] = normal_map
    return output


def depth_to_normal_sobel(view, depth):
    is_batch = True
    if depth.dim() == 3:
        is_batch = False
        depth = depth.unsqueeze(0)

    # Sobel kernels
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=depth.device) / 4
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=depth.device) / 4

    kernel_x = kernel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    kernel_y = kernel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    points = depths_to_points(view, depth)  # (B, H, W, 3)
    points_t = points.permute(0, 3, 1, 2)  # (B, 3, H, W)

    # 分别对X,Y,Z三个通道应用Sobel算子
    grad_x_channels = F.conv2d(points_t, kernel_x, padding='same', groups=3)
    grad_y_channels = F.conv2d(points_t, kernel_y, padding='same', groups=3)

    dx = grad_x_channels.permute(0, 2, 3, 1)
    dy = grad_y_channels.permute(0, 2, 3, 1)

    normal_map = torch.nn.functional.normalize(torch.cross(dy, dx, dim=-1), dim=-1)

    return normal_map if is_batch else normal_map.squeeze(0)
