#!/usr/bin/env python
"""
Run Phase 5 only after Phase 2 models are already available.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data_loader import load_and_clean
from src.sensitivity import run_phase5
from src.plot_sensitivity import plot_all_phase5


def main():
    df = load_and_clean()
    results = run_phase5(df)
    plot_all_phase5(results)
    return results


if __name__ == "__main__":
    main()
