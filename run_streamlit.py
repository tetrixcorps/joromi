import os
import torch
import psutil
import streamlit.web.bootstrap as bootstrap
from pathlib import Path

def monitor_resources():
    """Monitor system resources"""
    print("\nSystem Resources:")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")
    
    if torch.cuda.is_available():
        print("\nGPU Resources:")
        print(f"VRAM Used: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"VRAM Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

def cleanup_gpu():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def main():
    # Set environment variables
    os.environ["PYTHONPATH"] = str(Path(__file__).parent)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    
    # Initial cleanup
    cleanup_gpu()
    monitor_resources()
    
    try:
        # Run Streamlit
        print("\nStarting Streamlit server...")
        bootstrap.run(
            "frontend/app.py",
            "run",
            [],
            flag_options={
                "server.port": 8501,
                "server.address": "0.0.0.0"
            }
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
        cleanup_gpu()
        monitor_resources()

if __name__ == "__main__":
    main() 