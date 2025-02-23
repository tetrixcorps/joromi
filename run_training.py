#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from scripts.finetune_translation import main

if __name__ == "__main__":
    # Set up environment variables if needed
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Run training
    main() 