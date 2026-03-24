#!/usr/bin/env python3
"""
模型权重转移脚本
- 从124M GPT-2 small转移到355M GPT-2 medium
- 用于初始化训练
"""

import os
import torch
from transformers import GPT2LMHeadModel, GPT2Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transfer_weights():
    """转移权重"""
    logger.info("Starting weight transfer...")
    
    # 加载现有模型
    existing_model_dir = "./models/final_model_20260115_044955"
    if not os.path.exists(existing_model_dir):
        logger.error("Existing model directory not found!")
        return
    
    small_model = GPT2LMHeadModel.from_pretrained(existing_model_dir)
    small_state_dict = small_model.state_dict()
    
    # 创建GPT-2 medium配置
    medium_config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=1024,    # 1024 vs small的768
        n_layer=24,     # 24 vs small的12
        n_head=16,      # 16 vs small的12
        n_inner=4096,   # 4096 vs small的3072
    )
    
    # 创建medium模型
    medium_model = GPT2LMHeadModel(medium_config)
    medium_state_dict = medium_model.state_dict()
    
    # 转移匹配的权重
    transferred = 0
    for key in small_state_dict.keys():
        if key in medium_state_dict and small_state_dict[key].shape == medium_state_dict[key].shape:
            medium_state_dict[key] = small_state_dict[key]
            transferred += 1
            logger.info(f"Transferred: {key}")
    
    logger.info(f"Total transferred parameters: {transferred}")
    
    # 保存初始化的medium模型
    output_dir = "./models/medium_initialized"
    os.makedirs(output_dir, exist_ok=True)
    medium_model.save_pretrained(output_dir)
    
    logger.info(f"Initialized medium model saved to: {output_dir}")

if __name__ == "__main__":
    transfer_weights()
