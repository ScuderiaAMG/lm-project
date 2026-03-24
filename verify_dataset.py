#!/usr/bin/env python3
"""
数据集验证脚本 - 检查WikiText数据集是否可正常加载
"""

import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_wikitext_datasets():
    """验证不同WikiText数据集版本"""
    datasets_to_try = [
        ("wikitext", "wikitext-103-raw-v1"),
        ("wikitext", "wikitext-2-raw-v1"),
        ("wikitext", "wikitext-103-v1"),
        ("wikitext", "wikitext-2-v1"),
    ]
    
    for dataset_name, config_name in datasets_to_try:
        try:
            logger.info(f"Trying to load {dataset_name} with config {config_name}...")
            dataset = load_dataset(
                dataset_name,
                config_name,
                cache_dir="./data",
                trust_remote_code=True
            )
            
            logger.info(f"✓ Successfully loaded {dataset_name} ({config_name})")
            logger.info(f"Available splits: {list(dataset.keys())}")
            for split in dataset.keys():
                logger.info(f"  {split}: {len(dataset[split]):,} samples")
            
            return dataset_name, config_name, dataset
        
        except Exception as e:
            logger.warning(f"✗ Failed to load {dataset_name} ({config_name}): {str(e)}")
    
    logger.error("✗ All dataset attempts failed!")
    return None, None, None

if __name__ == "__main__":
    dataset_name, config_name, dataset = verify_wikitext_datasets()
    
    if dataset:
        logger.info("✓ Dataset verification successful!")
        logger.info(f"Using dataset: {dataset_name} with config: {config_name}")
    else:
        logger.error("✗ Dataset verification failed. Please check your internet connection and try again.")
        exit(1)
