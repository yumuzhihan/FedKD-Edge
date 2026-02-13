import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.teacher_cnn import TeacherCNN
from src.models.feature_adapter import FeatureAdapter
from src.trainers.base_trainer import BaseTrainer
from src.utils.get_logger import LoggerFactory


class CachedKDTrainer(BaseTrainer):
    def __init__(self, config, device, client_id, train_loader, num_classes):
        super().__init__(config, device, client_id, train_loader, num_classes)

        self.teacher = TeacherCNN(num_classes=num_classes).to(device)
        self.teacher.eval()
        self.is_cached = False

        strategy = config.get("strategy", "fedavg")
        s_ch = config.get("student_channels", 64)
        t_ch = config.get("teacher_channels", 128)

        self.adapter: nn.Module

        if strategy in ["feature_kd", "hybrid_kd"]:
            self.adapter = FeatureAdapter(
                in_channels=s_ch,
                out_channels=t_ch,
                target_spatial_size=(2, 2),
            ).to(device)

            self.optimizer.add_param_group({"params": self.adapter.parameters()})
        else:
            self.adapter = nn.Identity()

        self.logger = LoggerFactory.get_logger("Cached KD Trainer")

    def load_weights(self, global_state_dict, extra_payload: dict | None = None):
        super().load_weights(global_state_dict, extra_payload)
        if extra_payload and "teacher_state" in extra_payload:
            self.teacher.load_state_dict(extra_payload["teacher_state"])
            self._precompute_and_cache()
        else:
            raise ValueError(f"Strategy {self.config['strategy']} 需要 teacher_state")

    def _precompute_and_cache(self):
        if self.is_cached:
            return

        strategy = self.config["strategy"]
        self.teacher.eval()

        cpu_imgs, cpu_lbls = [], []
        cpu_t_logits, cpu_t_feats = [], []

        need_feat = strategy in ["feature_kd", "hybrid_kd"]
        need_logit = strategy in ["logit_kd", "hybrid_kd"]

        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                logits, features = self.teacher(images, return_features=True)

                if need_feat:
                    cpu_t_feats.append(features.cpu())
                if need_logit:
                    cpu_t_logits.append(logits.cpu())
                cpu_imgs.append(images.cpu())
                cpu_lbls.append(labels.cpu())

        data_tensors = [torch.cat(cpu_imgs), torch.cat(cpu_lbls)]

        if strategy == "logit_kd":
            data_tensors.append(torch.cat(cpu_t_logits))

        elif strategy == "feature_kd":
            data_tensors.append(torch.cat(cpu_t_feats))

        elif strategy == "hybrid_kd":
            data_tensors.append(torch.cat(cpu_t_logits))
            data_tensors.append(torch.cat(cpu_t_feats))

        self.train_loader = DataLoader(
            TensorDataset(*data_tensors),
            batch_size=self.config["local_bs"],
            shuffle=True,
        )

        self.is_cached = True
        self.teacher.cpu()

        del self.teacher
        torch.cuda.empty_cache()

    def compute_loss(self, *args):
        images, labels = args[0], args[1]
        strategy = self.config["strategy"]

        s_feat: torch.Tensor | None = None
        s_feat_proj: torch.Tensor | None = None
        if strategy in ["feature_kd", "hybrid_kd"]:
            s_logits, s_feat = self.model(images, return_features=True)
            s_feat_proj = self.adapter(s_feat)
        else:
            s_logits = self.model(images, return_features=False)

        loss_dict = {"outputs": s_logits}

        if strategy == "logit_kd":
            t_logits = args[2]
            loss, ce_loss, kd_loss = self._loss_kd_logit(s_logits, labels, t_logits)
            loss_dict.update(
                {
                    "loss": loss,
                    "ce_loss": ce_loss,
                    "kd_loss": kd_loss,
                }
            )
        elif strategy == "feature_kd":
            t_feat = args[2]
            if s_feat_proj is None:
                raise ValueError("s_feat_proj is None")
            loss, ce_loss, feat_loss = self._loss_kd_feature(
                s_logits, labels, s_feat_proj, t_feat
            )
            loss_dict.update(
                {
                    "loss": loss,
                    "ce_loss": ce_loss,
                    "feat_loss": feat_loss,
                }
            )
        elif strategy == "hybrid_kd":
            if s_feat_proj is None:
                raise ValueError("s_feat_proj is None")

            T = self.config["kd_T"]
            alpha = self.config["kd_alpha"]
            beta = self.config["hybrid_bata"]

            t_logits = args[2]
            t_feat = args[3]
            ce_loss = F.cross_entropy(s_logits, labels)
            kd_loss = F.kl_div(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(t_logits / T, dim=1),
                reduction="batchmean",
            ) * (T * T)
            fd_loss = F.mse_loss(s_feat_proj, t_feat)
            distillation_part = beta * kd_loss + (1 - beta) * fd_loss
            loss = (1 - alpha) * ce_loss + alpha * distillation_part
            loss_dict.update(
                {
                    "loss": loss,
                    "ce_loss": ce_loss,
                    "kd_loss": kd_loss,
                    "fd_loss": fd_loss,
                    "distillation_part": distillation_part,
                }
            )
        else:
            loss = F.cross_entropy(s_logits, labels)
            loss_dict["loss"] = loss
            loss_dict["ce_loss"] = loss

        return loss_dict

    def _loss_kd_logit(self, s_logits, labels, t_logits):
        T = self.config["kd_T"]
        alpha = self.config["kd_alpha"]

        kd_loss = F.kl_div(
            F.log_softmax(s_logits / T, dim=1),
            F.softmax(t_logits / T, dim=1),
            reduction="batchmean",
        ) * (T * T)

        ce_loss = F.cross_entropy(s_logits, labels)
        total_loss = (1 - alpha) * ce_loss + alpha * kd_loss
        return total_loss, ce_loss, kd_loss

    def _loss_kd_feature(self, s_logits, labels, s_feat_proj, t_feat):
        ce_loss = F.cross_entropy(s_logits, labels)

        feat_loss = F.mse_loss(s_feat_proj, t_feat)
        feat_alpha = self.config["feat_alpha"]
        total_loss = ce_loss + feat_alpha * feat_loss

        return total_loss, ce_loss, feat_loss
