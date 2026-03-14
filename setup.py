from setuptools import setup, find_packages

setup(
    name="nhl-pipeline",
    version="1.0.0",
    description="NHL 2025-2026 season data pipeline",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=[
        "nhl-api-py>=3.2.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
    entry_points={
        "console_scripts": [
            "nhl-pipeline=run_pipeline:main",
        ],
    },
)
