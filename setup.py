from setuptools import setup, find_packages
import os

# Read version from VERSION file
version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
with open(version_file, 'r') as f:
    version = f.read().strip()

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="slm-builder",
    version=version,
    author="SLM Builder Team",
    author_email="",
    description="Build Small/Specialized Language Models from any dataset, source, or topic",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/isathish/slm",
    project_urls={
        "Bug Tracker": "https://github.com/isathish/slm/issues",
        "Documentation": "https://github.com/isathish/slm/wiki",
        "Source Code": "https://github.com/isathish/slm",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "peft>=0.4.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.40.0",
        "structlog>=23.1.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
    ],
    extras_require={
        "full": [
            "sqlalchemy>=2.0.0",
            "psycopg2-binary>=2.9.0",
            "pymongo>=4.0.0",
            "pymysql>=1.0.0",
            "requests>=2.28.0",
            "tqdm>=4.65.0",
            "nltk>=3.8.0",
            "rouge-score>=0.1.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "streamlit>=1.25.0",
        ],
        "database": [
            "sqlalchemy>=2.0.0",
            "psycopg2-binary>=2.9.0",
            "pymongo>=4.0.0",
            "pymysql>=1.0.0",
        ],
        "api": [
            "requests>=2.28.0",
            "tqdm>=4.65.0",
        ],
        "metrics": [
            "nltk>=3.8.0",
            "rouge-score>=0.1.0",
        ],
        "serving": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "streamlit>=1.25.0",
        ],
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "mypy>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "slm=slm_builder.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
