import logging
from pathlib import Path
import datetime


class LoggerFactory:
    _loggers = {}

    @classmethod
    def get_logger(cls, name: str, log_level: int = logging.DEBUG) -> logging.Logger:
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(log_level)

            # 添加控制台处理器
            ch = logging.StreamHandler()
            ch.setLevel(log_level)

            # 添加文件处理器
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = (
                Path(__file__).parent.parent / "logs" / f"{name}_{current_time}.log"
            )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(file_path)
            fh.setLevel(log_level)

            # 设置日志格式
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)

            logger.addHandler(ch)
            logger.addHandler(fh)
            cls._loggers[name] = logger

        return cls._loggers[name]
