#!/usr/bin/env python3
"""
GPT-2 Medium (355M参数)训练脚本
- 8-bit量化优化
- 梯度检查点
- 针对RTX 4060 8GB显存优化
- 使用WikiText-103数据集
"""

import os
import time
import logging
from datetime import datetime
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    set_seed
)
from datasets import load_dataset
from rich.console import Console
import warnings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
console = Console()

# 设置随机种子
set_seed(42)

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, message="`resume_download` is deprecated")

def setup_environment():
    """设置环境变量和性能优化"""
    logger.info("Setting up environment...")
    
    # 禁用Hugging Face在线检查
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    logger.info("Offline mode enabled")
    
    # 性能优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        logger.info(f"GPU current memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    return device

def load_and_preprocess_data():
    """加载和预处理WikiText-103数据集"""
    logger.info("Loading and preprocessing WikiText-103 dataset...")
    
    try:
        # 从本地缓存加载数据集
        dataset = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            cache_dir="./data",
            trust_remote_code=True
        )
        
        logger.info(f"Dataset loaded successfully!")
        logger.info(f"Train samples: {len(dataset['train']):,}")
        logger.info(f"Validation samples: {len(dataset['validation']):,}")
        
        # 加载tokenizer
        logger.info("Loading GPT-2 Medium tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained(
            "gpt2-medium",
            local_files_only=True,
            cache_dir="./models"
        )
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")
        
        # Tokenize函数
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=1024,  # 增加序列长度
                return_tensors="pt"
            )
        
        # 应用tokenize
        logger.info("Tokenizing dataset...")
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text", "id"] if "id" in dataset["train"].column_names else ["text"],
            desc="Tokenizing",
            load_from_cache_file=True,
            cache_file_names={
                "train": "./data/train_wikitext103_tokenized.cache",
                "validation": "./data/validation_wikitext103_tokenized.cache"
            }
        )
        
        # 设置格式为PyTorch
        for split in ["train", "validation"]:
            if split in tokenized_datasets:
                tokenized_datasets[split] = tokenized_datasets[split].remove_columns(
                    [col for col in tokenized_datasets[split].column_names if col != "input_ids"]
                )
                tokenized_datasets[split].set_format(type="torch", columns=["input_ids"])
        
        return tokenized_datasets, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        logger.info("Falling back to WikiText-2 dataset...")
        return load_wikitext2_dataset()

def load_wikitext2_dataset():
    """回退到WikiText-2数据集"""
    logger.warning("Using WikiText-2 dataset as fallback")
    
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        cache_dir="./data",
        trust_remote_code=True
    )
    
    logger.info(f"WikiText-2 dataset loaded successfully!")
    logger.info(f"Train samples: {len(dataset['train']):,}")
    logger.info(f"Validation samples: {len(dataset['validation']):,}")
    
    tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2-medium", 
        local_files_only=True,
        cache_dir="./models"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing WikiText-2"
    )
    
    for split in ["train", "validation"]:
        if split in tokenized_datasets:
            tokenized_datasets[split] = tokenized_datasets[split].remove_columns(
                [col for col in tokenized_datasets[split].column_names if col != "input_ids"]
            )
            tokenized_datasets[split].set_format(type="torch", columns=["input_ids"])
    
    return tokenized_datasets, tokenizer

def create_model():
    """创建GPT-2 Medium模型 (355M参数) - 8-bit量化"""
    logger.info("Creating GPT-2 Medium model (355M parameters) with 8-bit quantization...")
    
    try:
        # 8-bit量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_use_double_quant=True,
        )
        
        logger.info("Loading 8-bit quantized GPT-2 Medium model...")
        model = GPT2LMHeadModel.from_pretrained(
            "gpt2-medium",
            quantization_config=bnb_config,
            local_files_only=True,
            cache_dir="./models",
            device_map="auto"  # 自动设备映射
        )
        
        # 启用梯度检查点
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # 打印模型信息
        total_params = model.num_parameters()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Memory footprint: {model.get_memory_footprint() / 1024**3:.2f} GB")
        
        return model
    
    except Exception as e:
        logger.error(f"Error loading 8-bit model: {str(e)}")
        logger.info("Falling back to standard GPT-2 Medium model...")
        return GPT2LMHeadModel.from_pretrained("gpt2-medium").cuda()

def main():
    """主训练函数"""
    console.rule("[bold blue]GPT-2 Medium Training (355M Parameters)")
    start_time = time.time()
    
    try:
        # 设置环境
        device = setup_environment()
        
        # 加载数据
        tokenized_datasets, tokenizer = load_and_preprocess_data()
        
        # 创建模型
        model = create_model()
        model.to(device)
        
        # 配置训练参数 - 针对8-bit优化
        training_args = TrainingArguments(
            output_dir="./results_medium",
            overwrite_output_dir=True,
            num_train_epochs=10,  # 增加训练轮数
            per_device_train_batch_size=1,  # 减小batch size适应更大模型
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32,  # 32 × 1 = 32有效batch size
            learning_rate=3e-5,  # 降低学习率
            weight_decay=0.01,
            fp16=True,  # 混合精度
            logging_steps=50,  # 更频繁的日志
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            dataloader_num_workers=4,
            optim="paged_adamw_8bit",  # 8-bit优化器
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            save_total_limit=2,
            max_grad_norm=1.0,
            logging_dir="./logs_medium",
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
        )
        
        # 开始训练
        console.rule("[bold green]Starting Training")
        logger.info("Training configuration:")
        logger.info(f"- Model: GPT-2 Medium (355M parameters)")
        logger.info(f"- Dataset: WikiText-103")
        logger.info(f"- Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        logger.info(f"- Total epochs: {training_args.num_train_epochs}")
        logger.info(f"- 8-bit quantization: Enabled")
        logger.info(f"- Gradient checkpointing: Enabled")
        
        train_result = trainer.train()
        
        # 保存结果
        console.rule("[bold green]Training Completed")
        metrics = train_result.metrics
        
        # 保存最终模型
        final_model_dir = f"./models/final_model_medium_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(final_model_dir, exist_ok=True)
        
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        # 保存训练指标
        with open(os.path.join(final_model_dir, "training_metrics.txt"), "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Final model saved to: {final_model_dir}")
        logger.info(f"Training metrics: {metrics}")
        
        # 生成示例文本
        generate_example_text(model, tokenizer, device, final_model_dir)
        
        # 总时间
        total_time = time.time() - start_time
        logger.info(f"Total training time: {total_time/3600:.2f} hours")
        logger.info(f"Training speed: {metrics.get('train_samples_per_second', 0):.2f} samples/second")
        
        return final_model_dir
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

def generate_example_text(model, tokenizer, device, output_dir):
    """生成示例文本并保存"""
    logger.info("Generating example text...")
    
    try:
        model.eval()
        prompts = [
            "The future of artificial intelligence is",
            "Machine learning has revolutionized",
            "Natural language processing allows computers to",
            "In the next decade, technology will",
            "Shandong jianzhu univerisity is known for"
        ]
        
        generated_texts = []
        
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,  # 增加生成长度
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
            logger.info(f"\nPrompt: '{prompt}'")
            logger.info(f"Generated: '{generated_text}'")
        
        # 保存生成的文本
        with open(os.path.join(output_dir, "generated_examples.txt"), "w") as f:
            for i, text in enumerate(generated_texts):
                f.write(f"Example {i+1}:\n{text}\n\n")
        
        logger.info("Generated examples saved successfully!")
        
    except Exception as e:
        logger.warning(f"Failed to generate example text: {str(e)}")

if __name__ == "__main__":
    main()
