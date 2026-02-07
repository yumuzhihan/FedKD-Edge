import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.trainers.base_trainer import BaseTrainer
from src.models.teacher_cnn import TeacherCNN


class LogitKDTrainer(BaseTrainer):
    def __init__(self, config, device, client_id, train_loader):
        super().__init__(config, device, client_id, train_loader)

        self.teacher = TeacherCNN().to(device)
        self.teacher.eval()
        self.teacher_logits = None

    def load_weights(self, global_state_dict, extra_payload: dict | None = None):
        super().load_weights(global_state_dict)

        if extra_payload is not None and "teacher_state" in extra_payload:
            self.teacher.load_state_dict(extra_payload["teacher_state"])
        else:
            raise ValueError("LogitKDTrainer需要在extra_payload中传入teacher_state")

    def compute_loss(self, *args):
        images, labels = args[0], args[1]
        outputs = self.model(images)
        if self.teacher_logits is None:
            with torch.no_grad():
                self.teacher_logits = self.teacher(images)

        total_loss, ce_loss, kd_loss = self.loss_fn_kd(
            outputs, labels, self.teacher_logits
        )

        return {
            "loss": total_loss,
            "outputs": outputs,
            "ce_loss": ce_loss,
            "kd_loss": kd_loss,
        }

    def loss_fn_kd(self, outputs, labels, teacher_outputs):
        temp = self.config["kd_T"]
        alpha = self.config["kd_alpha"]

        kd_loss = F.kl_div(
            F.log_softmax(outputs / temp, dim=1),
            F.softmax(teacher_outputs / temp, dim=1),
            reduction="batchmean",
        ) * (temp * temp)

        ce_loss = F.cross_entropy(outputs, labels)
        total_loss = ce_loss * (1 - alpha) + kd_loss * alpha

        return total_loss, ce_loss, kd_loss
