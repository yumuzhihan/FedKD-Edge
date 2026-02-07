from torch import nn
from src.utils.get_logger import LoggerFactory


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class StudentCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, log_output=False):
        super(StudentCNN, self).__init__()

        self.logger = LoggerFactory.get_logger(self.__class__.__name__)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(120, num_classes),
        )

        self._initialize_weights()
        if log_output:
            self.logger.info(f"模型初始化完成，参数数量: {count_parameters(self)}")

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        feat = self.conv2(x)
        x = self.classifier(feat)

        if return_features:
            return x, feat
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
