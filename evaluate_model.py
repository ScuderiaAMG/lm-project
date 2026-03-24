#!/usr/bin/env python3
"""
离线模式：模型评估脚本
- 禁用Hugging Face在线检查
- 使用本地缓存的数据集
- 计算困惑度和生成质量
"""

import os
import torch
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_offline_mode():
    """启用离线模式"""
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    logger.info("Offline mode enabled - no internet connection required")

def load_model_and_tokenizer():
    """加载模型和tokenizer (离线模式)"""
    logger.info("Loading model and tokenizer from cache...")
    
    model_dir = "./models/final_model_20260115_044955"
    
    try:
        # 从本地加载tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(
            model_dir,
            local_files_only=True
        )
        
        # 从本地加载模型
        model = GPT2LMHeadModel.from_pretrained(
            model_dir,
            local_files_only=True
        ).cuda()
        
        logger.info("Model and tokenizer loaded successfully!")
        logger.info(f"Model parameters: {model.num_parameters():,}")
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error("Please check if the model directory exists and contains the correct files.")
        raise

def load_validation_dataset():
    """加载验证数据集 (离线模式)"""
    logger.info("Loading validation dataset from cache...")
    
    try:
        # 从本地缓存加载WikiText-2验证集
        dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="validation",
            cache_dir="./data",
            trust_remote_code=True
        )
        
        logger.info(f"Validation dataset loaded successfully!")
        logger.info(f"Number of validation samples: {len(dataset)}")
        
        return dataset
    
    except Exception as e:
        logger.error(f"Error loading validation dataset: {str(e)}")
        logger.error("Falling back to a small sample of training data...")
        
        # 回退到训练数据集的小样本
        dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train[:100]",  # 使用前100个训练样本
            cache_dir="./data",
            trust_remote_code=True
        )
        
        logger.info("Using training data subset as validation fallback")
        return dataset

def calculate_perplexity(model, tokenizer, dataset):
    """计算困惑度"""
    logger.info("Calculating perplexity...")
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    processed_samples = 0
    
    # 只评估前50个样本
    for i, example in enumerate(dataset):
        if i >= 50:  # 限制评估样本数量
            break
            
        text = example["text"]
        if len(text) < 10:  # 跳过太短的文本
            continue
        
        try:
            # Tokenize文本
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to("cuda")
            
            # 计算损失
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["input_ids"]
                )
                loss = outputs.loss
            
            # 累积损失和token数
            num_tokens = inputs["input_ids"].size(1)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            processed_samples += 1
            
            if i % 10 == 0:
                logger.info(f"Processed {i+1}/50 samples...")
                
        except Exception as e:
            logger.warning(f"Error processing sample {i}: {str(e)}")
            continue
    
    if total_tokens == 0:
        logger.error("No valid samples processed for perplexity calculation")
        return float('inf')
    
    # 计算平均损失和困惑度
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    logger.info(f"Processed {processed_samples} samples")
    logger.info(f"Average loss: {avg_loss:.4f}")
    logger.info(f"Perplexity: {perplexity:.2f}")
    
    return perplexity

def test_text_generation(model, tokenizer):
    """测试文本生成质量"""
    logger.info("\nTesting text generation quality...")
    
    prompts = [
        "The future of artificial intelligence is",
        "Machine learning has revolutionized",
        "Natural language processing allows computers to",
        "In the next decade, technology will",
        "Artificial intelligence is transforming"
    ]
    
    results = []
    
    for prompt in prompts:
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append((prompt, generated_text))
            
            logger.info(f"\nPrompt: '{prompt}'")
            logger.info(f"Generated: '{generated_text}'")
            
        except Exception as e:
            logger.warning(f"Error generating text for prompt '{prompt}': {str(e)}")
    
    return results

def main():
    """主评估函数"""
    logger.info("=== Starting Model Evaluation (Offline Mode) ===")
    
    try:
        # 设置离线模式
        setup_offline_mode()
        
        # 加载模型和tokenizer
        model, tokenizer = load_model_and_tokenizer()
        
        # 加载验证数据集
        validation_dataset = load_validation_dataset()
        
        # 计算困惑度
        perplexity = calculate_perplexity(model, tokenizer, validation_dataset)
        
        # 测试文本生成
        generation_results = test_text_generation(model, tokenizer)
        
        # 保存评估结果
        results_file = "./models/final_model_20260115_044955/evaluation_results.txt"
        with open(results_file, "w") as f:
            f.write("=== Model Evaluation Results ===\n")
            f.write(f"Perplexity: {perplexity:.2f}\n")
            f.write("\nGeneration Samples:\n")
            for prompt, generated_text in generation_results:
                f.write(f"\nPrompt: '{prompt}'\n")
                f.write(f"Generated: '{generated_text}'\n")
        
        logger.info(f"\nEvaluation completed successfully!")
        logger.info(f"Perplexity: {perplexity:.2f}")
        logger.info(f"Results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
