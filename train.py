import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.trainer1 import train
import torch
import subprocess
...

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch device in use:", torch.cuda.get_device_name(torch.cuda.current_device()))
    train(device)

