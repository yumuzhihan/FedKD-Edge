import sys

sys.path.append("..")

import torch
from torch import nn
from utils.get_logger import LoggerFactory


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TeacherCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
    ) -> None:
        super(TeacherCNN, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, padding=1, bias=False):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=bias,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        self.logger = LoggerFactory.get_logger(self.__class__.__name__)

        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

        self._initialize_weights()
        self.logger.info(f"模型初始化完成，参数数量: {count_parameters(self)}")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
