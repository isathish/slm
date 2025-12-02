---
layout: default
title: Developer Guide
nav_order: 8
---

# Developer Guide

This guide covers everything you need to know to contribute to and develop SLM Builder.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Release Process](#release-process)
- [Testing](#testing)
- [Code Style](#code-style)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda

### Clone the Repository

```bash
git clone https://github.com/isathish/slm.git
cd slm
```

### Development Setup

1. **Create a virtual environment:**

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n slm python=3.8
conda activate slm
```

2. **Install development dependencies:**

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Or install from requirements
pip install -r requirements-dev.txt
```

3. **Install pre-commit hooks:**

```bash
pre-commit install
```

## 📁 Project Structure

```
slm/
├── slm_builder/               # Main package
│   ├── __init__.py
│   ├── models/                # Model implementations
│   │   ├── base.py
│   │   ├── qa_model.py
│   │   └── transformer_model.py
│   ├── data/                  # Data loaders
│   │   ├── base_loader.py
│   │   ├── database_loader.py
│   │   ├── api_loader.py
│   │   └── file_loader.py
│   ├── training/              # Training utilities
│   │   ├── trainer.py
│   │   └── optimizer.py
│   ├── evaluation/            # Evaluation metrics
│   │   ├── metrics.py
│   │   └── evaluator.py
│   └── utils/                 # Utility functions
│       ├── quantization.py
│       ├── config.py
│       └── helpers.py
├── tests/                     # Test suite
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── examples/                  # Example scripts
├── docs/                      # Documentation
├── .github/                   # GitHub workflows
│   └── workflows/
│       ├── auto-release.yml
│       ├── release.yml
│       ├── jekyll-gh-pages.yml
│       └── tests.yml
├── VERSION                    # Version file
├── setup.py
├── pyproject.toml
└── README.md
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/slm.git
cd slm
git remote add upstream https://github.com/isathish/slm.git
```

### 2. Create a Branch

Use descriptive branch names:

```bash
# For features
git checkout -b feat/add-new-loader

# For bug fixes
git checkout -b fix/resolve-memory-leak

# For documentation
git checkout -b docs/update-api-reference
```

### 3. Make Your Changes

- Write clear, concise code
- Add tests for new features
- Update documentation
- Follow the code style guidelines

### 4. Commit Your Changes

Use conventional commit messages:

```bash
# Features
git commit -m "feat: Add PostgreSQL data loader"

# Bug fixes
git commit -m "fix: Resolve model loading timeout"

# Documentation
git commit -m "docs: Update installation instructions"

# Performance improvements
git commit -m "perf: Optimize tokenization process"

# Refactoring
git commit -m "refactor: Simplify database connection logic"

# Tests
git commit -m "test: Add tests for API loader"

# Breaking changes
git commit -m "BREAKING CHANGE: Remove deprecated build_from_text() method"
```

**Commit Prefix Reference:**

| Prefix | Type | Release Impact |
|--------|------|----------------|
| `feat:` | New feature | Minor version bump |
| `fix:` | Bug fix | Patch version bump |
| `docs:` | Documentation | Patch version bump |
| `perf:` | Performance | Patch version bump |
| `refactor:` | Code refactoring | Patch version bump |
| `test:` | Tests | Patch version bump |
| `chore:` | Maintenance | Patch version bump |
| `BREAKING CHANGE:` | Breaking change | Major version bump |

### 5. Push and Create Pull Request

```bash
git push origin your-branch-name
```

Then create a pull request on GitHub:

1. Go to your fork on GitHub
2. Click "Pull Request"
3. Provide a clear title and description
4. Link related issues if applicable

### Pull Request Guidelines

- **Title**: Use conventional commit format
- **Description**: Explain what and why (not how)
- **Tests**: Ensure all tests pass
- **Documentation**: Update docs if needed
- **Small PRs**: Keep changes focused and atomic

## 🔄 Release Process

SLM Builder uses automated semantic versioning based on commit messages.

### Automatic Releases (Recommended)

Releases are automatically created when you push to main:

```bash
# This will create a minor release (e.g., 1.0.0 → 1.1.0)
git commit -m "feat: Add new evaluation metric"
git push origin main

# This will create a patch release (e.g., 1.1.0 → 1.1.1)
git commit -m "fix: Resolve tokenization issue"
git push origin main

# This will create a major release (e.g., 1.1.1 → 2.0.0)
git commit -m "BREAKING CHANGE: Change API interface"
git push origin main
```

**How it works:**
1. Auto-release workflow analyzes commit messages
2. Determines version bump type (major/minor/patch)
3. Updates VERSION file
4. Creates git tag (e.g., v1.2.3)
5. Release workflow generates release notes
6. GitHub release is created automatically

### Manual Releases

If you need to create a release manually:

**Option 1: Using the script**

```bash
./create-release.sh
```

Follow the interactive prompts to select version bump type.

**Option 2: Manual tag**

```bash
# Create and push a tag
git tag -a v1.2.3 -m "Release v1.2.3"
git push origin v1.2.3
```

**Option 3: GitHub Actions UI**

1. Go to Actions → Create Release
2. Click "Run workflow"
3. Enter version and release type

### Version Strategy

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (x.0.0): Breaking changes
- **MINOR** (x.y.0): New features, backward compatible
- **PATCH** (x.y.z): Bug fixes, backward compatible

### What Gets Released

- Python package to PyPI (if configured)
- GitHub release with auto-generated notes
- Updated CHANGELOG.md
- Documentation updates

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=slm_builder --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py

# Run specific test
pytest tests/unit/test_models.py::test_model_loading

# Run with verbose output
pytest -v
```

### Writing Tests

Place tests in the `tests/` directory:

```python
# tests/unit/test_loader.py
import pytest
from slm_builder.data import DatabaseLoader

def test_database_loader_connection():
    """Test database connection establishment."""
    loader = DatabaseLoader(host="localhost", port=5432)
    assert loader.is_connected()

def test_database_loader_query():
    """Test database query execution."""
    loader = DatabaseLoader(host="localhost", port=5432)
    results = loader.query("SELECT * FROM users LIMIT 10")
    assert len(results) <= 10

@pytest.mark.parametrize("db_type", ["postgresql", "mysql", "sqlite"])
def test_multiple_databases(db_type):
    """Test support for multiple database types."""
    loader = DatabaseLoader(db_type=db_type)
    assert loader.db_type == db_type
```

### Test Coverage Goals

- Minimum 80% code coverage
- 100% coverage for critical paths
- Test both success and failure cases
- Include integration tests for key features

## 🎨 Code Style

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Imports**: Use absolute imports
- **Docstrings**: Google style

### Formatting Tools

```bash
# Format code with black
black slm_builder/ tests/

# Check formatting
black --check slm_builder/ tests/

# Sort imports
isort slm_builder/ tests/

# Lint with flake8
flake8 slm_builder/ tests/

# Type checking with mypy
mypy slm_builder/
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

### Docstring Example

```python
def load_data(source: str, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    Load data from a source and return a DataLoader.
    
    Args:
        source: Path to data source or database connection string
        batch_size: Number of samples per batch (default: 32)
        shuffle: Whether to shuffle the data (default: True)
    
    Returns:
        DataLoader instance configured with the specified parameters
    
    Raises:
        FileNotFoundError: If source file doesn't exist
        ConnectionError: If database connection fails
    
    Example:
        >>> loader = load_data("data/train.json", batch_size=64)
        >>> for batch in loader:
        ...     process_batch(batch)
    """
    pass
```

## 📝 Documentation

### Building Documentation Locally

```bash
# Install Jekyll and dependencies
bundle install

# Serve documentation locally
cd docs
bundle exec jekyll serve

# View at http://localhost:4000/slm/
```

### Documentation Guidelines

- Use clear, concise language
- Include code examples
- Add screenshots for UI features
- Keep documentation up-to-date with code changes
- Use Markdown format
- Add front matter to new pages:

```yaml
---
layout: default
title: Your Page Title
nav_order: 5
---
```

### Documentation Structure

- **README.md**: Quick overview and getting started
- **INSTALLATION.md**: Installation instructions
- **FEATURES.md**: Feature documentation
- **EXAMPLES.md**: Code examples
- **DEVELOPER_GUIDE.md**: This file
- **API Reference**: Auto-generated from docstrings

## 🐛 Debugging

### Common Issues

**Import Errors:**
```bash
# Ensure package is installed in editable mode
pip install -e .
```

**Test Failures:**
```bash
# Clear pytest cache
pytest --cache-clear

# Run tests in verbose mode
pytest -v -s
```

**Documentation Build Errors:**
```bash
# Clear Jekyll cache
cd docs
bundle exec jekyll clean
bundle exec jekyll build
```

### Debugging Tools

```python
# Use pdb for debugging
import pdb; pdb.set_trace()

# Or use ipdb (enhanced debugger)
import ipdb; ipdb.set_trace()

# Logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

## 🔗 Additional Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## 💬 Getting Help

- **Issues**: [GitHub Issues](https://github.com/isathish/slm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/isathish/slm/discussions)
- **Documentation**: [https://isathish.github.io/slm/](https://isathish.github.io/slm/)

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

Thank you for contributing to SLM Builder! 🚀
