import logging
from pathlib import Path
import datetime
import time
from zoneinfo import ZoneInfo


class LoggerFactory:
    _loggers = {}
    _file_name = (
        datetime.datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H") + ".log"
    )

    @classmethod
    def get_logger(cls, name: str, log_level: int = logging.DEBUG) -> logging.Logger:
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(log_level)

            # 添加控制台处理器
            ch = logging.StreamHandler()
            ch.setLevel(log_level)

            # 添加文件处理器
            log_dir = Path(__file__).parent.parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            file_path = log_dir / f"{cls._file_name}"

            # 清理旧日志，最多保留10份
            log_files = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime)
            while len(log_files) >= 10:
                if len(log_files) == 10 and file_path.exists():
                    break
                oldest_file = log_files.pop(0)
                if oldest_file.name == file_path.name:
                    continue
                try:
                    oldest_file.unlink()
                except Exception:
                    pass

            fh = logging.FileHandler(file_path)
            fh.setLevel(log_level)

            # 设置日志格式
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            formatter.converter = time.gmtime
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)

            logger.addHandler(ch)
            logger.addHandler(fh)
            cls._loggers[name] = logger

        return cls._loggers[name]
