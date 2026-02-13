from typing import Any
from torch import optim
from torch.amp import grad_scaler, autocast_mode
from torch.utils.tensorboard import SummaryWriter
import os

from src.models.student_cnn import StudentCNN


class BaseTrainer:
    """
    联邦学习训练器基类
    """

    def __init__(self, config, device, client_id, train_loader, num_classes):
        self.config = config
        self.device = device
        self.client_id = client_id

        self.train_loader = train_loader

        self.model = StudentCNN(num_classes=num_classes).to(device)
        self.model.train()

        self.optimizer = optim.SGD(
            self.model.parameters(), lr=config["lr"], momentum=config["momentum"]
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["local_ep"]
        )
        self.scaler = grad_scaler.GradScaler()

        # TensorBoard writer
        log_dir = os.path.join(config.get("log_dir", "runs"), f"client_{client_id}")
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

    def load_weights(self, global_state_dict, extra_payload: dict | None = None):
        self.model.load_state_dict(global_state_dict)

    def compute_loss(self, *args) -> dict[str, Any]:
        """
        计算损失函数

        Returns:
            dict: 包含以下键的字典
                - 'loss': 总损失值 (torch.Tensor)
                - 'outputs': 模型输出 (torch.Tensor)
                - 其他可选的损失组件用于监控
        """
        raise NotImplementedError("子类必须实现compute_loss方法进行损失计算")

    def train(self):
        epoch_loss = []
        epoch_acc = []

        for epoch in range(self.config["local_ep"]):
            batch_loss = []
            correct = 0
            total = 0

            for batch in self.train_loader:
                batch = [t.to(self.device, non_blocking=True) for t in batch]
                labels = batch[1]
                self.optimizer.zero_grad()

                with autocast_mode.autocast(
                    device_type=self.device.type, enabled=(self.device.type == "cuda")
                ):
                    loss_dict = self.compute_loss(*batch)

                loss = loss_dict["loss"]
                outputs = loss_dict["outputs"]

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                batch_loss.append(loss.item())
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # 记录到 TensorBoard (每个 batch)
                self.writer.add_scalar("Loss/batch", loss.item(), self.global_step)

                # 记录其他损失组件
                for key, value in loss_dict.items():
                    if key not in ["loss", "outputs"] and hasattr(value, "item"):
                        self.writer.add_scalar(
                            f"Loss/{key}", value.item(), self.global_step
                        )

                self.global_step += 1

            self.scheduler.step()

            avg_loss = sum(batch_loss) / len(batch_loss) if batch_loss else 0.0
            acc = 100.0 * correct / total if total > 0 else 0.0
            epoch_loss.append(avg_loss)
            epoch_acc.append(acc)

            # 记录每个 epoch 的统计信息
            self.writer.add_scalar("Loss/epoch", avg_loss, epoch)
            self.writer.add_scalar("Accuracy/epoch", acc, epoch)
            self.writer.add_scalar(
                "Learning_Rate", self.optimizer.param_groups[0]["lr"], epoch
            )

        return self.get_upload_package(), epoch_loss, epoch_acc

    def get_upload_package(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
