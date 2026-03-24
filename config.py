"""
配置文件 - 所有参数集中管理
"""
import os

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")

# 确保目录存在
for dir_path in [DATA_DIR, MODEL_DIR, LOG_DIR, OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 模型配置
MODEL_CONFIG = {
    "base_model": "gpt2",           # 基础模型
    "max_length": 512,              # 最大序列长度
    "batch_size": 4,                # 每设备batch size
    "gradient_accumulation_steps": 8,  # 梯度累积步数
    "learning_rate": 5e-5,          # 学习率
    "num_epochs": 5,                # 训练轮数
    "warmup_ratio": 0.1,            # 预热比例
}

# 硬件配置
HARDWARE_CONFIG = {
    "fp16": True,                   # 混合精度
    "gradient_checkpointing": True, # 梯度检查点
    "num_workers": 4,               # 数据加载worker数
    "pin_memory": True,             # 锁页内存
}

# 数据集配置
DATASET_CONFIG = {
    "name": "wikitext",
    "config_name": "wikitext-103-raw-v1",
    "train_split": "train",
    "validation_split": "validation",
    "cache_dir": DATA_DIR,
}

# 训练配置
TRAINING_CONFIG = {
    "output_dir": OUTPUT_DIR,
    "logging_dir": LOG_DIR,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 100,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "report_to": "none",  # 不使用wandb
}

# 未来扩展配置
SCALING_CONFIG = {
    "enable_larger_models": False,  # 启用更大模型
    "use_8bit_optimization": False, # 8-bit优化
    "distributed_training": False,  # 分布式训练
}
