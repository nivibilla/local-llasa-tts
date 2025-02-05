#!/usr/bin/env python3
import sys
import argparse
import torch

# Check Python version
if sys.version_info < (3, 10):
    print("ERROR: Python 3.10 or higher is required.")
    sys.exit(1)

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available. Please use a CUDA-capable GPU.")
    sys.exit(1)

import os
from .inference import initialize_models
from .models import get_llasa_model
from .app import build_dashboard

def main():
    parser = argparse.ArgumentParser(description="Run the modular Llasa TTS Dashboard.")
    parser.add_argument("--share", help="Enable gradio share", action="store_true")
    args = parser.parse_args()

    print("Initializing CUDA backend...", flush=True)
    torch.cuda.init()
    _ = torch.zeros(1).cuda()
    print(f"Using device: {torch.cuda.get_device_name()}", flush=True)

    # Initialize local models
    print("\nStep 1: Loading XCodec2 and Whisper models...", flush=True)
    initialize_models()

    print("\nStep 2: Preloading Llasa 3B model (faster startup for standard usage)...", flush=True)
    get_llasa_model("3B")
    print("Preload done. Models are ready!")

    # Launch Gradio
    print("\nLaunching Gradio interface...", flush=True)
    app = build_dashboard()
    app.launch(share=args.share, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
