import torch.nn.functional as F
from src.trainers.base_trainer import BaseTrainer


class FedAvgTrainer(BaseTrainer):
    def __init__(self, config, device, client_id, train_loader, num_classes):
        super().__init__(config, device, client_id, train_loader, num_classes)

    def compute_loss(self, *args):
        images, labels = args[0], args[1]
        outputs = self.model(images)

        loss = F.cross_entropy(outputs, labels)

        return {
            "loss": loss,
            "outputs": outputs,
            "ce_loss": loss,
        }
