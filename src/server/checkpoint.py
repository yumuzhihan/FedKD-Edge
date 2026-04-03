"""
断点续训练检查点管理模块
"""

import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.server.worker import FederatedServer


@dataclass
class CheckpointState:
    """检查点状态数据结构"""

    round_idx: int  # 当前完成的轮次（0-indexed）
    w_glob: Dict[str, Any]  # 全局模型权重
    w_adapter: Optional[Dict[str, Any]] = None  # Adapter权重
    config: Dict[str, Any] = field(default_factory=dict)  # 配置信息
    csv_path: str = ""  # CSV日志文件路径
    timestamp: str = ""  # 检查点创建时间戳

    @classmethod
    def from_server(
        cls, server: "FederatedServer", round_idx: int
    ) -> "CheckpointState":
        """从 FederatedServer 实例创建检查点状态"""
        return cls(
            round_idx=round_idx,
            w_glob={k: v.cpu().clone() for k, v in server.w_glob.items()},
            w_adapter=(
                {k: v.cpu().clone() for k, v in server.w_adapter.items()}
                if server.w_adapter
                else None
            ),
            config=server.config.copy(),
            csv_path=str(server.csv_path),
            timestamp=time.strftime("%Y%m%d-%H%M%S"),
        )

    def save(self, path: Path):
        """保存检查点到文件"""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "CheckpointState":
        """从文件加载检查点"""
        with open(path, "rb") as f:
            return pickle.load(f)


class CheckpointManager:
    """检查点管理器"""

    CHECKPOINT_PATTERN = re.compile(
        r"checkpoint_(?P<strategy>\w+)_(?P<dataset>\w+)_(?P<partition_tag>.+)_seed(?P<seed>\d+)_round(?P<round>\d+)_\d+\.pth"
    )

    @staticmethod
    def experiment_checkpoint_name(
        strategy: str, dataset: str, seed: int, partition_tag: str
    ) -> str:
        return f"checkpoint_{strategy}_{dataset}_{partition_tag}_seed{seed}.pth"

    @staticmethod
    def final_checkpoint_name(
        strategy: str, dataset: str, seed: int, partition_tag: str
    ) -> str:
        return f"checkpoint_final_{strategy}_{dataset}_{partition_tag}_seed{seed}.pth"

    @staticmethod
    def find_latest_checkpoint(
        checkpoint_dir: Path,
        strategy: str,
        dataset: str,
        seed: int,
        partition_tag: str,
    ) -> Optional[Path]:
        """自动查找最新的匹配检查点"""
        latest_path = checkpoint_dir / CheckpointManager.experiment_checkpoint_name(
            strategy, dataset, seed, partition_tag
        )
        if latest_path.exists():
            return latest_path

        legacy_final_path = checkpoint_dir / CheckpointManager.final_checkpoint_name(
            strategy, dataset, seed, partition_tag
        )
        if legacy_final_path.exists():
            return legacy_final_path

        checkpoints = []
        for f in checkpoint_dir.glob("checkpoint_*.pth"):
            match = CheckpointManager.CHECKPOINT_PATTERN.match(f.name)
            if match:
                if (
                    match.group("strategy") == strategy
                    and match.group("dataset") == dataset
                    and match.group("partition_tag") == partition_tag
                    and int(match.group("seed")) == seed
                ):
                    checkpoints.append((int(match.group("round")), f))

        if not checkpoints:
            return None

        # 返回轮次最高的检查点
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints[-1][1]

    @staticmethod
    def validate_checkpoint(
        checkpoint: CheckpointState, config: Dict[str, Any]
    ) -> List[str]:
        """
        验证检查点与当前配置是否兼容。
        返回警告信息列表（空列表表示完全兼容）。
        """
        warnings = []

        # 关键字段必须完全匹配
        critical_fields = [
            "strategy",
            "dataset",
            "seed",
            "num_users",
            "frac",
            "partition_mode",
            "partition_seed",
            "client_classes",
        ]
        for field_name in critical_fields:
            checkpoint_val = checkpoint.config.get(field_name)
            current_val = config.get(field_name)
            if checkpoint_val != current_val:
                warnings.append(
                    f"Config mismatch in '{field_name}': "
                    f"checkpoint={checkpoint_val}, "
                    f"current={current_val}. Resume may produce unexpected results."
                )

        # 检查 rounds
        checkpoint_rounds = checkpoint.config.get("rounds", 0)
        current_rounds = config.get("rounds", 0)
        if checkpoint_rounds > current_rounds:
            warnings.append(
                f"Checkpoint was trained to {checkpoint_rounds} rounds, "
                f"but current config only requests {current_rounds} rounds. "
                f"Training will stop at round {current_rounds}."
            )
        elif checkpoint.round_idx >= current_rounds:
            warnings.append(
                f"Checkpoint already at round {checkpoint.round_idx + 1}, "
                f"which is >= requested rounds {current_rounds}. Nothing to do."
            )

        return warnings
