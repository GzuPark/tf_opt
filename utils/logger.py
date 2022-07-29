import logging
import os

from datetime import datetime


def get_logger(root_path: str, target: str) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    output_dir = os.path.join(root_path, "artifacts")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger_path = os.path.join(output_dir, f"{target}_{now}.log")

    file_formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    file_handler = logging.FileHandler(logger_path)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    stream_formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    return logger
