#!/usr/bin/env python3
"""
Language Model Training Script
- Uses Hugging Face Transformers and Datasets
- Optimized for RTX 4060 8GB VRAM
- Supports mixed precision and gradient checkpointing
- Complete logging and monitoring
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
    default_data_collator,
)
from datasets import load_dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
console = Console()

def setup_environment():
    """Setup environment variables and performance optimizations"""
    logger.info("Setting up environment...")
    
    # Performance optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    return device

def load_and_preprocess_data():
    """Load and preprocess dataset"""
    logger.info("Loading and preprocessing dataset...")
    
    try:
        # Load dataset from Hugging Face
        dataset = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            cache_dir="./data",
            trust_remote_code=True
        )
        
        logger.info(f"Dataset loaded successfully!")
        logger.info(f"Train samples: {len(dataset['train'])}")
        logger.info(f"Validation samples: {len(dataset['validation'])}")
        
        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
        
        # Apply tokenization
        logger.info("Tokenizing dataset...")
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing",
            load_from_cache_file=True
        )
        
        # Set format to PyTorch
        tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        return tokenized_datasets, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        logger.info("Using smaller dataset for testing...")
        return load_dummy_dataset()

def load_dummy_dataset():
    """Load a small dummy dataset for testing"""
    logger.warning("Loading dummy dataset for testing...")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create small test dataset
    small_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir="./data")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    tokenized_datasets = small_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dummy dataset"
    )
    
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized_datasets, tokenizer

def create_model():
    """Create and configure model"""
    logger.info("Creating GPT-2 model...")
    
    try:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
        
        logger.info(f"Model created with {model.num_parameters():,} parameters")
        return model
    
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        logger.info("Creating model from scratch...")
        return GPT2LMHeadModel.from_pretrained("gpt2")

def main():
    """Main training function"""
    console.rule("[bold blue]Language Model Training")
    start_time = time.time()
    
    try:
        # Setup environment
        device = setup_environment()
        
        # Load data
        tokenized_datasets, tokenizer = load_and_preprocess_data()
        
        # Create model
        model = create_model()
        model.to(device)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            overwrite_output_dir=True,
            num_train_epochs=5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=5e-5,
            weight_decay=0.01,
            fp16=True,  # Mixed precision
            logging_steps=100,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            dataloader_num_workers=4,
            optim="adamw_torch_fused",
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            save_total_limit=3,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=default_data_collator,
        )
        
        # Train
        console.rule("[bold green]Starting Training")
        logger.info("Training configuration:")
        logger.info(f"- Model: GPT-2 (124M parameters)")
        logger.info(f"- Batch size: 4 × 8 = 32 (effective)")
        logger.info(f"- Total epochs: 5")
        logger.info(f"- Output directory: ./results")
        
        train_result = trainer.train()
        
        # Save results
        console.rule("[bold green]Training Completed")
        metrics = train_result.metrics
        
        # Save final model
        final_model_dir = f"./models/final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(final_model_dir, exist_ok=True)
        
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        logger.info(f"Model saved to: {final_model_dir}")
        logger.info(f"Training metrics: {metrics}")
        
        # Total time
        total_time = time.time() - start_time
        logger.info(f"Total training time: {total_time/3600:.2f} hours")
        
        return final_model_dir
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
