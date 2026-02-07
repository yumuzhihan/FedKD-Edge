from torch import nn
import torch.nn.functional as F


class FeatureAdapter(nn.Module):
    def __init__(
        self, in_channels, out_channels, hidden_dim=None, target_spatial_size=None
    ):
        super(FeatureAdapter, self).__init__()

        if hidden_dim is None:
            hidden_dim = max(in_channels, out_channels)

        self.target_spatial_size = target_spatial_size

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        x = self.net(x)

        # If target spatial size is specified, adapt the spatial dimensions
        if self.target_spatial_size is not None:
            x = F.adaptive_avg_pool2d(x, self.target_spatial_size)

        return x
