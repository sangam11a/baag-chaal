"""
Pytest configuration and top-level environment initialization.
Ensures numpy and torch native binaries are initialized in correct sequence.
"""

import numpy as np
import torch

# Ensure deterministic test environment
np.random.seed(42)
torch.manual_seed(42)

