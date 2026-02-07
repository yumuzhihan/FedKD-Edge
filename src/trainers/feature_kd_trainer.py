import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from src.trainers.base_trainer import BaseTrainer
from src.models.teacher_cnn import TeacherCNN
from src.models.feature_adapter import FeatureAdapter


class FeatureKDTrainer(BaseTrainer):
    def __init__(self, config, device, client_id, train_loader):
        super().__init__(config, device, client_id, train_loader)
        self.teacher = TeacherCNN().to(device)
        self.feature_adapter = FeatureAdapter(
            in_channels=config["student_channels"],
            out_channels=config["teacher_channels"],
        ).to(device)

        self.teacher.eval()
        self.teacher.train()

        self.criterion_feat = nn.MSELoss(reduction="mean")

        self.optimizer = optim.SGD(
            list(self.model.parameters()) + list(self.feature_adapter.parameters()),
            lr=config["lr"],
            momentum=config["momentum"],
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["local_ep"]
        )

        self.t_feat = None

    def load_weights(self, global_state_dict, extra_payload: dict | None = None):
        super().load_weights(global_state_dict)

        if extra_payload is not None and "teacher_state" in extra_payload:
            self.teacher.load_state_dict(extra_payload["teacher_state"])
        else:
            raise ValueError("FeatureKDTrainer需要在extra_payload中传入teacher_state")

    def compute_loss(self, *args):
        images, labels = args[0], args[1]
        outputs, s_feat = self.model(images, return_features=True)

        if self.t_feat is None:
            with torch.no_grad():
                _, self.t_feat = self.teacher(images, return_features=True)

        s_feat_proj = self.feature_adapter(s_feat)

        loss_feat = self.criterion_feat(s_feat_proj, self.t_feat)
        loss_cls = F.cross_entropy(outputs, labels)

        feature_alpha = self.config["feat_alpha"]
        total_loss = feature_alpha * loss_feat + (1 - feature_alpha) * loss_cls

        return {
            "loss": total_loss,
            "outputs": outputs,
            "ce_loss": loss_cls,
            "feat_loss": loss_feat,
        }

    def get_upload_package(self):
        pkg = super().get_upload_package()
        pkg["feature_adapter"] = self.feature_adapter.state_dict()
        return pkg
