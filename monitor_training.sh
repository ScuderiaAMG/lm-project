#!/bin/bash
"""
训练监控脚本
- 监控GPU使用情况
- 查看训练日志
- 检查磁盘空间
"""

echo "=== Training Monitor ==="

# 监控GPU
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

echo -e "\nDisk Space:"
df -h "$HOME/lm-project"

echo -e "\nTraining Logs (last 20 lines):"
tail -n 20 ./logs/training.log 2>/dev/null || echo "No training log found yet"

echo -e "\nCheckpoints:"
ls -la ./results/checkpoint-* 2>/dev/null || echo "No checkpoints found yet"
