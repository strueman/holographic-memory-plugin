import os
import sys

# Add project root to path so 'store', 'retrieval', 'holographic' are importable
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
