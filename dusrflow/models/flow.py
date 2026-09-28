"""DuFlowNet wrapper, warping utilities, and training losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dusrflow.models.archs.flow_2E import Flow_Net


def backward_warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = image.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=image.device),
        torch.arange(width, device=image.device),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=0).float().unsqueeze(0).repeat(batch, 1, 1, 1)
    grid = grid + flow
    grid_x = 2.0 * grid[:, 0] / (width - 1) - 1.0
    grid_y = 2.0 * grid[:, 1] / (height - 1) - 1.0
    sampling_grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(
        image,
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


class DuFlowNet(nn.Module):
    """Predict LRC-to-Ref flow and backward-warp Ref into the LRC domain."""

    def __init__(self):
        super().__init__()
        self.flow_net = Flow_Net()

    def forward(self, lr_center: torch.Tensor, reference: torch.Tensor):
        flow = self.flow_net(lr_center, reference)
        warped_reference = backward_warp(reference, flow)
        return warped_reference, flow


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        return torch.sqrt((prediction - target) ** 2 + self.epsilon).mean()


class EdgeAwareSecondOrderSmoothness(nn.Module):
    def __init__(self, alpha: float = 10.0):
        super().__init__()
        self.alpha = alpha

    @staticmethod
    def gradient(tensor: torch.Tensor):
        dx = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        dy = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
        return dx, dy

    def forward(self, flow: torch.Tensor, image: torch.Tensor):
        image_dx, image_dy = self.gradient(image)
        weight_x = torch.exp(-image_dx.abs().mean(1, keepdim=True) * self.alpha)
        weight_y = torch.exp(-image_dy.abs().mean(1, keepdim=True) * self.alpha)
        flow_dx, flow_dy = self.gradient(flow)
        flow_dxx, _ = self.gradient(flow_dx)
        _, flow_dyy = self.gradient(flow_dy)
        loss_x = (weight_x[:, :, :, 1:] * flow_dxx.abs()).mean()
        loss_y = (weight_y[:, :, 1:, :] * flow_dyy.abs()).mean()
        return 0.5 * (loss_x + loss_y)
