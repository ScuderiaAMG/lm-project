#!/bin/bash
# Language model training startup script
# - Activate conda environment
# - Set environment variables
# - Run training
# - Handle interruptions and resumption

# Activate conda environment
source ~/anaconda3/bin/activate
conda activate lm-training

# Check if environment is activated successfully
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Error: Conda environment not activated properly"
    echo "Please run: conda activate lm-training"
    exit 1
fi

echo "=== Conda environment activated: $CONDA_DEFAULT_ENV ==="

# Set project directory
PROJECT_DIR="$HOME/lm-project"
cd "$PROJECT_DIR"

# Create necessary directories
mkdir -p ./data ./results ./models ./logs

# Run training
echo "=== Starting GPT language model training ==="
echo "Training will run for approximately 5 days"
echo "Press Ctrl+C to interrupt training (will save checkpoint)"

# Run training script
python train_gpt.py

# Check training result
if [ $? -eq 0 ]; then
    echo "=== Training completed successfully ==="
    echo "Final model saved in the models directory"
    echo "Generated examples available in the model directory"
else
    echo "=== Training interrupted or failed ==="
    echo "Check the logs in the logs directory for details"
    echo "You can resume training by running this script again"
fi
