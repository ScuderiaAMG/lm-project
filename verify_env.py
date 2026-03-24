import sys
import torch
import platform

print("=== Environment Verification ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")

print(f"\nConda environment: {sys.prefix}")

# 检查PyTorch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("\n✓ Environment verification successful!")
