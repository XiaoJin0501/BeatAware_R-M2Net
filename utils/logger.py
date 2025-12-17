import logging
import os
import sys

def setup_logger(log_dir, name="train"):
    """
    配置 Logger：同时输出到控制台和文件
    Args:
        log_dir: 日志保存目录
        name: 日志文件名 (不含扩展名)
    """
    # 1. 创建 Logger 对象
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False # 防止重复打印

    # 如果已经有 Handler (避免重复添加)
    if logger.handlers:
        return logger

    # 2. 定义格式
    # 格式示例: [2023-10-25 10:00:00] INFO: Epoch 1 started...
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 3. 输出到文件 (FileHandler)
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.FileHandler(file_path, mode='a') # 'a' 表示追加模式
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 4. 输出到屏幕 (StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger