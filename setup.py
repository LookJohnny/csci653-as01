"""
DarkHorse - AI-Driven Demand Forecasting & Load Balancing
Setup configuration for pip installation
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="darkhorse-forecasting",
    version="1.0.0",
    author="DarkHorse Team",
    description="AI-driven demand forecasting and load balancing for high-throughput distributed systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/darkhorse",
    packages=find_packages(exclude=["tests", "tests.*"]),
    py_modules=[
        "amazon_unify_pipeline",
        "build_weekly_dataset",
        "config",
        "dataCleaning",
        "forecast_pipeline",
        "generate_asin_mapping",
        "generate_dq_report",
        "load_balancer",
        "main",
        "train_transformer",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "darkhorse=main:main",
            "darkhorse-unify=amazon_unify_pipeline:main",
            "darkhorse-build-weekly=build_weekly_dataset:main",
            "darkhorse-train=train_transformer:main",
            "darkhorse-forecast=forecast_pipeline:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
    ],
    keywords="forecasting machine-learning time-series load-balancing transformer autots",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/darkhorse/issues",
        "Source": "https://github.com/yourusername/darkhorse",
    },
)