#!/usr/bin/env python3
"""Setup for holographic-memory-plugin standalone testing."""

from setuptools import setup, find_packages

setup(
    name="holographic-memory-plugin",
    version="0.1.0",
    description="Holographic memory plugin for Hermes Agent — SQLite-backed fact storage with entity resolution, trust scoring, HRR vectors, and RRF retrieval.",
    author="Simon (strueman)",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[],  # stdlib-only core; optional deps in requirements.txt
    extras_require={
        "vec": ["sqlite-vec", "onnxruntime", "tokenizers"],
        "hrr": ["numpy"],
        "test": ["pytest>=7.0"],
        "all": ["sqlite-vec", "onnxruntime", "tokenizers", "numpy", "pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [],
    },
)
