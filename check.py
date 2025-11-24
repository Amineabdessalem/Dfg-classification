import torch
print("CUDA available (GPU available):", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current CUDA device: (GPU number)", torch.cuda.current_device())
    print("Device name: (GPU name)", torch.cuda.get_device_name(torch.cuda.current_device()))