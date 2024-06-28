import torch

# 检查 PyTorch 版本
print(f"PyTorch version: {torch.__version__}")

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
else:
    print("CUDA is not available.")