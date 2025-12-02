# Changelog

## [v1.0.0] - 2025-12-02

### 🚨 Major Release
This is a major release with significant new features and potential breaking changes.

### 📊 Release Statistics
- **Commits**: 13 total
- **Release Type**: major
- **Date**: December 02, 2025

### ✨ New Features
- feat: Enhance release workflow with comprehensive release notes generation and update documentation structure (4e3bdde)
- feat: Add comprehensive documentation and release setup for SLM Builder (0e11277)
- feat: Add comprehensive examples and completion documentation; update contributing guidelines (000f1d8)
- feat: Add database loaders for SQL and MongoDB (d37e1a9)
- feat: Enhance model loading and evaluation capabilities (067540b)
- feat: Implement FastAPI server for SLM model serving (6a02e24)

### 🐛 Bug Fixes
- fix: Correct import order for model export and quantization functions (82b6296)

### 📚 Documentation
- feat: Enhance release workflow with comprehensive release notes generation and update documentation structure (4e3bdde)

### ♻️ Code Refactoring
- refactor: Simplify list comprehensions and sorting in model comparison (c050aca)
- refactor: Clean up unused imports and improve code readability across multiple files (37e742e)

### 🔧 Maintenance
- chore: Remove obsolete GitHub Actions workflow for publishing documentation to Wiki (2e3a3c0)


layout: default
title: Changelog
nav_order: 9
---

# Changelog

All notable changes to the SLM Builder project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and workflows

## [1.0.0] - 2025-12-02

### Added
- Core SLM Builder functionality
- Multiple data source support (CSV, JSONL, text, URLs)
- Database loaders (PostgreSQL, MySQL, SQLite, MongoDB)
- REST API data loader with authentication and pagination
- Dynamic model loading from multiple sources
- Model zoo with 20+ pre-configured models
- LoRA and full fine-tuning support
- Quantization (4-bit, 8-bit) support
- Dataset splitting with stratification
- K-fold cross-validation
- Dataset validation and quality checks
- Model comparison and benchmarking
- Experiment tracking system
- Evaluation metrics (Perplexity, Accuracy, BLEU, ROUGE, F1)
- Export formats (ONNX, TorchScript, HuggingFace)
- Comprehensive documentation
- Example scripts
- CLI interface
- FastAPI serving template

### Documentation
- Core features guide (FEATURES.md)
- Additional features guide (ADDITIONAL_FEATURES.md)
- Comprehensive examples guide (EXAMPLES.md)
- Contributing guidelines (CONTRIBUTING.md)
- Development checklist (CHECKLIST.md)
- Complete API documentation

### Infrastructure
- GitHub Actions workflows for documentation publishing
- Automated release workflow with semantic versioning
- Version bump workflow
- Wiki publishing pipeline
- Issue and PR templates

---

## Version History

### Semantic Versioning Guidelines

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version (X.0.0): Incompatible API changes
- **MINOR** version (0.X.0): New functionality in a backward compatible manner
- **PATCH** version (0.0.X): Backward compatible bug fixes

### Release Types

- **Major Release**: Breaking changes, major new features, API redesign
- **Minor Release**: New features, improvements, no breaking changes
- **Patch Release**: Bug fixes, documentation updates, minor improvements

---

[Unreleased]: https://github.com/isathish/slm/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/isathish/slm/releases/tag/v1.0.0
