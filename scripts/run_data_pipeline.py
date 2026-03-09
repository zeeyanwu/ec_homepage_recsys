#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script will be the main entry point for the data processing pipeline.
It will load a configuration file and trigger the corresponding ETL steps.
"""

import argparse
import os
import sys

# Add project root to path to allow absolute imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.utils.config_loader import load_config
from src.pipeline.data_pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the data processing pipeline.")
    parser.add_argument('--config', type=str, required=True, help='Path to the data configuration YAML file.')
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config)

if __name__ == '__main__':
    main()
