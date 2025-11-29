#!/usr/bin/env python3
"""
run_b2t.py - Wrapper script for running the BrainToText training.

This script provides a convenient entry point for running the training,
especially when called from Modal or other orchestration systems.
"""

import sys
import os

# Ensure the repository root is in the Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def main(debug=False, train=True):
    """
    Run the BrainToText training.

    Args:
        debug: If True, load only a subset of data for faster iteration.
        train: If True, train the model; otherwise, load existing weights and evaluate.
    """
    # Import main module here to ensure path is set up correctly
    import main as b2t_main
    b2t_main.main(debug=debug, train=train)


if __name__ == "__main__":
    # Default: run in debug mode for quick testing
    main(debug=True, train=True)
