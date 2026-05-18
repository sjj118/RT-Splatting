import torch


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, -1, keepdim=True)


def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2 * dot(x, n) * n - x


def length(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))  # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN


def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / length(x, eps)


def linear2srgb(tensor_0to1):
    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor_0to1 = torch.clamp(tensor_0to1, min=1e-10, max=1.0)

    tensor_linear = tensor_0to1 * srgb_linear_coeff
    tensor_nonlinear = srgb_exponential_coeff * (torch.pow(tensor_0to1, 1 / srgb_exponent)) - (srgb_exponential_coeff - 1)

    is_linear = tensor_0to1 <= srgb_linear_thres
    tensor_srgb = torch.where(is_linear, tensor_linear, tensor_nonlinear)
    return tensor_srgb


def srgb2linear(tensor_0to1):
    srgb_linear_thres = 0.04045
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor_0to1 = torch.clamp(tensor_0to1, min=1e-10, max=1.0)

    tensor_linear = tensor_0to1 / srgb_linear_coeff
    tensor_nonlinear = torch.pow(tensor_0to1 + (srgb_exponential_coeff - 1) / srgb_exponential_coeff, srgb_exponent)

    is_linear = tensor_0to1 <= srgb_linear_thres
    tensor_srgb = torch.where(is_linear, tensor_linear, tensor_nonlinear)
    return tensor_srgb
