import sys
import platform

print("--- System Information ---")
print(f"Python Version: {sys.version}")
print(f"Operating System: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")

# If your lab involves machine learning (e.g., PyTorch), use this:
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Available compute device: {device.upper()}")
except ImportError:
    print("PyTorch not installed. Assuming CPU device for general Python.")
    print("Available compute device: CPU")

print("--------------------------")
