# src/trainers/fedkd_hybrid_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.models.teacher_cnn import TeacherCNN
from src.models.feature_adapter import FeatureAdapter
from src.trainers.base_trainer import BaseTrainer


class FedKDHybridTrainer(BaseTrainer):
    def __init__(self, config, device, client_id, train_loader, num_classes):
        # 1. 先调用父类初始化 (这会初始化 model 和默认 optimizer)
        super().__init__(config, device, client_id, train_loader,num_classes)

        # 2. 初始化 Teacher (冻结)
        self.teacher = TeacherCNN().to(device)
        self.teacher.eval()

        # 3. 初始化 Adapter (可训练)
        self.adapter = FeatureAdapter(
            config["student_channels"], config["teacher_channels"]
        ).to(device)
        self.adapter.train()

        # 4. [关键] 重新初始化优化器
        # 因为父类的 optimizer 只包含了 self.model.parameters()
        # 这里我们需要它包含 self.adapter.parameters()
        self.optimizer = optim.SGD(
            list(self.model.parameters()) + list(self.adapter.parameters()),
            lr=config["lr"],
            momentum=config["momentum"],
        )

        # 5. 重新绑定 Scheduler 到新的优化器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["local_ep"]
        )

        # Feature Loss 函数
        self.criterion_feat = nn.MSELoss(reduction="mean")

        # Teacher 缓存
        self.t_outputs = None
        self.t_feat = None

    def load_weights(self, global_state_dict, extra_payload: dict | None = None):
        super().load_weights(global_state_dict, extra_payload)

        if (
            extra_payload
            and "teacher_state" in extra_payload
            and "adapter_state" in extra_payload
        ):
            self.teacher.load_state_dict(extra_payload["teacher_state"])
            self.adapter.load_state_dict(extra_payload["adapter_state"])
        else:
            raise ValueError("Hybrid需要同时传入Teacher Model和Adapter")

    def compute_loss(self, *args):
        images, labels = args[0], args[1]
        outputs, s_feat = self.model(images, return_features=True)

        if self.t_outputs is None or self.t_feat is None:
            with torch.no_grad():
                self.t_outputs, self.t_feat = self.teacher(images, return_features=True)

        s_feat_proj = self.adapter(s_feat)

        loss_task = F.cross_entropy(outputs, labels)

        T = self.config["kd_T"]
        alpha_logit = self.config["kd_alpha"]

        loss_logit = F.kl_div(
            F.log_softmax(outputs / T, dim=1),
            F.softmax(self.t_outputs / T, dim=1),
            reduction="batchmean",
        ) * (T * T)

        alpha_feat = self.config["feat_alpha"]  # Feature KD 的权重
        loss_feat = self.criterion_feat(s_feat_proj, self.t_feat)

        total_loss = (
            loss_task * (1 - alpha_logit)
            + loss_logit * alpha_logit
            + loss_feat * alpha_feat
        )

        return {
            "loss": total_loss,
            "outputs": outputs,
            "ce_loss": loss_task,
            "kd_loss": loss_logit,
            "feat_loss": loss_feat,
        }

    def get_upload_package(self):
        pkg = super().get_upload_package()
        pkg["adapter"] = {k: v.cpu() for k, v in self.adapter.state_dict().items()}
        return pkg
