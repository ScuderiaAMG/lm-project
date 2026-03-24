#!/usr/bin/env python3
"""
下载GPT-2 Medium模型和tokenizer
"""

import os
import logging
from transformers import GPT2Tokenizer, GPT2LMHeadModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    """下载GPT-2 Medium模型"""
    logger.info("Downloading GPT-2 Medium model (355M parameters)...")
    
    # 创建模型目录
    os.makedirs("./models", exist_ok=True)
    
    # 下载tokenizer
    logger.info("Downloading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.save_pretrained("./models/gpt2-medium")
    
    # 下载模型
    logger.info("Downloading model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    model.save_pretrained("./models/gpt2-medium")
    
    logger.info("GPT-2 Medium model and tokenizer downloaded successfully!")
    logger.info("You can now run training in offline mode.")

if __name__ == "__main__":
    download_model()
