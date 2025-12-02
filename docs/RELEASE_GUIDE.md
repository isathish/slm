# Documentation Organization & Release Guide

## 📁 Documentation Structure

All documentation has been moved to the `docs/` directory:

```
docs/
├── README.md                    # Documentation index
├── FEATURES.md                  # Core features guide
├── ADDITIONAL_FEATURES.md       # Advanced features guide
├── EXAMPLES.md                  # Comprehensive examples guide
├── CONTRIBUTING.md              # Contributing guidelines
├── CHECKLIST.md                 # Development checklist
├── IMPLEMENTATION_SUMMARY.md    # Implementation details
├── COMPLETION_REPORT.md         # Project completion status
└── TODO_COMPLETION.md           # Completed tasks
```

## 📖 GitHub Wiki Publishing

Documentation is automatically published to the GitHub Wiki when changes are pushed to the `main` branch.

### How Wiki Publishing Works

1. **Automatic Publishing**: The `.github/workflows/publish-wiki.yml` workflow runs on every push to `main` that modifies files in the `docs/` directory.

2. **Wiki Structure**:
   - `docs/FEATURES.md` → Wiki page: `Features`
   - `docs/ADDITIONAL_FEATURES.md` → Wiki page: `Additional-Features`
   - `docs/EXAMPLES.md` → Wiki page: `Examples`
   - `docs/CONTRIBUTING.md` → Wiki page: `Contributing`
   - `docs/CHECKLIST.md` → Wiki page: `Development-Checklist`
   - Auto-generated `Home` page with navigation

3. **Manual Publishing**: You can also trigger the wiki publishing workflow manually from the GitHub Actions tab.

### Accessing the Wiki

View the wiki at: `https://github.com/isathish/slm/wiki`

## 🚀 Release Workflow

The project uses semantic versioning with automated release workflows.

### Semantic Versioning

Versions follow the format: `MAJOR.MINOR.PATCH`

- **MAJOR** (1.0.0): Breaking changes, major new features
- **MINOR** (0.1.0): New features, backward compatible
- **PATCH** (0.0.1): Bug fixes, minor improvements

### Creating a Release

#### Option 1: Using Version Bump Workflow (Recommended)

1. Go to GitHub Actions → "Version Bump" workflow
2. Click "Run workflow"
3. Select bump type (major/minor/patch)
4. Choose whether to create a release
5. Click "Run workflow"

This will:
- Calculate the new version number
- Update version in all files (`VERSION`, `setup.py`, `__init__.py`)
- Create a git commit with the version bump
- Create and push a version tag
- Optionally trigger the release workflow

#### Option 2: Manual Tag Push

1. Update version manually:
   ```bash
   # Update VERSION file
   echo "1.0.0" > VERSION
   
   # Update __init__.py
   # Change __version__ = "1.0.0"
   
   # Commit changes
   git add VERSION slm_builder/__init__.py
   git commit -m "🔖 Bump version to 1.0.0"
   git push
   ```

2. Create and push tag:
   ```bash
   git tag -a v1.0.0 -m "Release 1.0.0"
   git push origin v1.0.0
   ```

3. The release workflow will automatically trigger.

### What the Release Workflow Does

1. **Detects version** from tag (e.g., `v1.0.0` → `1.0.0`)
2. **Determines release type** (major/minor/patch)
3. **Generates release notes** with:
   - Version information
   - Installation instructions
   - Feature highlights
   - Documentation links
4. **Builds Python package** (wheel and source distribution)
5. **Creates GitHub Release** with:
   - Release notes
   - Distribution files
   - Auto-generated changelog
6. **Comments on commit** with release link

### Release Notes

Release notes are automatically generated and include:
- Release type badge (MAJOR/MINOR/PATCH)
- Installation instructions
- Feature highlights
- Documentation links
- Changelog

### Version Tracking

Current version is stored in three places:
- `VERSION` file (source of truth)
- `slm_builder/__init__.py` (`__version__` variable)
- `setup.py` (reads from VERSION file)

## 📝 Changelog Management

Update `CHANGELOG.md` before creating a release:

1. Add new version section:
   ```markdown
   ## [1.0.0] - 2025-12-02
   
   ### Added
   - New feature description
   
   ### Changed
   - Changed feature description
   
   ### Fixed
   - Bug fix description
   ```

2. Update links at bottom:
   ```markdown
   [1.0.0]: https://github.com/isathish/slm/releases/tag/v1.0.0
   ```

## 🔄 Workflow Files

### 1. `publish-wiki.yml`
- **Trigger**: Push to `main` with changes in `docs/`
- **Purpose**: Publish documentation to GitHub Wiki
- **Manual Run**: Yes, via Actions tab

### 2. `release.yml`
- **Trigger**: Push of version tag (e.g., `v1.0.0`)
- **Purpose**: Create GitHub release with artifacts
- **Manual Run**: Yes, specify version and release type

### 3. `version-bump.yml`
- **Trigger**: Manual workflow dispatch
- **Purpose**: Bump version and optionally create release
- **Manual Run**: Yes, choose major/minor/patch

## 📊 Current Version

**Version**: 1.0.0 (Initial Release)

## 🎯 Release Checklist

Before creating a release:

- [ ] All tests pass
- [ ] Code formatted (black, isort)
- [ ] No linting errors (flake8)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in all files
- [ ] Examples tested
- [ ] README.md reviewed

## 🐛 Issue Templates

Issue templates are available for:
- **Bug Reports**: `.github/ISSUE_TEMPLATE/bug_report.md`
- **Feature Requests**: `.github/ISSUE_TEMPLATE/feature_request.md`

## 🔀 Pull Request Template

PR template is available at: `.github/PULL_REQUEST_TEMPLATE.md`

Includes checklist for:
- Code quality
- Testing
- Documentation
- Style compliance

## 📚 Additional Resources

- [Semantic Versioning Guide](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Wiki Documentation](https://docs.github.com/en/communities/documenting-your-project-with-wikis)

---

**Last Updated**: December 2, 2025
