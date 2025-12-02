# Contributing to SLM-Builder

Thank you for your interest in contributing to SLM-Builder! This document provides guidelines for contributing to the project.

## 🚀 Quick Start

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/slm.git
   cd slm
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[full,dev]"
   ```

## 🔨 Development Workflow

### Setting Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[full,dev]"

# Run tests to verify setup
pytest tests/
```

### Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Format code:
   ```bash
   black slm_builder tests
   isort slm_builder tests
   ```

4. Run tests:
   ```bash
   pytest tests/ -v
   ```

5. Check linting:
   ```bash
   flake8 slm_builder tests --max-line-length=100
   ```

### Commit Guidelines

We follow conventional commit messages:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or modifications
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

Example:
```bash
git commit -m "feat: add support for Parquet data loader"
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=slm_builder --cov-report=html

# Run specific test file
pytest tests/test_loaders.py -v
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive test names
- Include docstrings explaining what is tested

Example:
```python
def test_csv_loader_with_custom_columns():
    """Test CSV loader with custom column mapping."""
    # Test implementation
    pass
```

## 📝 Documentation

### Docstring Format

We use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Short description of function.
    
    Longer description if needed, explaining behavior,
    edge cases, etc.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
    """
    pass
```

### Type Hints

Always include type hints for function parameters and return values:

```python
from typing import List, Dict, Optional

def process_data(
    records: List[Dict[str, Any]],
    config: Optional[Config] = None
) -> List[Dict[str, Any]]:
    pass
```

## 🎯 Areas for Contribution

### High Priority
- [ ] Database connectors (PostgreSQL, MongoDB)
- [ ] Additional export formats
- [ ] More evaluation metrics
- [ ] Streaming dataset support
- [ ] Better error messages

### Medium Priority
- [ ] Additional model architectures
- [ ] Advanced quantization methods
- [ ] Experiment tracking integration
- [ ] Web UI for training monitoring

### Good First Issues
- [ ] Documentation improvements
- [ ] Example scripts
- [ ] Bug fixes
- [ ] Test coverage improvements

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment**:
   - OS and version
   - Python version
   - Package version
   - Installation method

2. **Description**:
   - What you expected to happen
   - What actually happened
   - Steps to reproduce

3. **Code Example**:
   ```python
   # Minimal reproducible example
   ```

4. **Error Message**:
   ```
   Full error traceback
   ```

## 💡 Feature Requests

For feature requests, please provide:

1. **Use Case**: What problem does this solve?
2. **Proposed Solution**: How would it work?
3. **Alternatives**: What alternatives have you considered?
4. **Examples**: Mock code showing desired usage

## 📋 Pull Request Process

1. **Update Documentation**: Update README, docstrings, etc.
2. **Add Tests**: Ensure new code has test coverage
3. **Update Changelog**: Add entry to CHANGELOG.md
4. **Pass CI**: All tests and linting must pass
5. **Request Review**: Tag maintainers for review

### PR Checklist

- [ ] Tests pass locally
- [ ] Code formatted with black and isort
- [ ] Linting passes (flake8)
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Commits follow conventional format

## 🏗️ Project Structure

```
slm_builder/
├── data/           # Data loading and preprocessing
├── models/         # Model training and export
├── serve/          # Serving utilities
├── utils/          # Helper utilities
├── api.py          # Main API
└── cli.py          # CLI interface
```

## 🤝 Code Review Process

1. Maintainers review PRs within 1-2 weeks
2. Address review comments
3. Once approved, maintainer will merge
4. PR author should be responsive to feedback

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Thanked in the README

## 📧 Questions?

- Open a GitHub Discussion
- Check existing issues
- Read the documentation

Thank you for contributing to SLM-Builder! 🎉
