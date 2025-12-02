# 🚀 Automatic Release Guide

## How It Works

The auto-release workflow automatically detects version bumps from your commit messages and creates releases.

## 📝 Commit Message Format

Use these prefixes in your commit messages to trigger automatic releases:

### Major Release (Breaking Changes)
Triggers version: **x.0.0**

```bash
git commit -m "BREAKING CHANGE: Remove deprecated API"
git commit -m "major: Complete API redesign"
```

**Use when:**
- Breaking API changes
- Removing features
- Major refactors that break compatibility

### Minor Release (New Features)
Triggers version: **x.y.0**

```bash
git commit -m "feat: Add MongoDB support"
git commit -m "feature: Add new evaluation metrics"
git commit -m "✨ Add API data loader"
```

**Use when:**
- Adding new features
- New functionality
- New data sources

### Patch Release (Bug Fixes)
Triggers version: **x.y.z**

```bash
git commit -m "fix: Resolve database connection issue"
git commit -m "bugfix: Fix memory leak"
git commit -m "🐛 Fix import error"
git commit -m "patch: Update dependencies"
```

**Use when:**
- Bug fixes
- Performance improvements
- Documentation updates
- Dependency updates
- Maintenance tasks

## 🔄 Workflow Process

1. **You push to main** with a commit message
2. **Auto-release workflow runs**:
   - Analyzes commit messages
   - Determines bump type (major/minor/patch)
   - Calculates new version
   - Updates VERSION file
   - Creates git tag (e.g., v1.2.3)
   - Pushes tag to GitHub
3. **Release workflow triggers** (on tag creation):
   - Generates comprehensive release notes
   - Categorizes changes (features, fixes, etc.)
   - Builds packages
   - Creates GitHub release
   - Updates CHANGELOG.md

## 📋 Examples

### Example 1: Adding a Feature

```bash
git add slm_builder/data/redis_loader.py
git commit -m "feat: Add Redis data loader with authentication support"
git push origin main
```

**Result**: Minor version bump (e.g., 1.0.0 → 1.1.0)

### Example 2: Fixing a Bug

```bash
git add slm_builder/models/base.py
git commit -m "fix: Resolve model loading timeout issue"
git push origin main
```

**Result**: Patch version bump (e.g., 1.1.0 → 1.1.1)

### Example 3: Breaking Change

```bash
git add slm_builder/api.py
git commit -m "BREAKING CHANGE: Remove deprecated build_from_text() method"
git push origin main
```

**Result**: Major version bump (e.g., 1.1.1 → 2.0.0)

### Example 4: Multiple Changes

```bash
# Commit 1
git commit -m "feat: Add PostgreSQL support"

# Commit 2
git commit -m "feat: Add model comparison feature"

# Commit 3
git commit -m "fix: Resolve connection issue"

git push origin main
```

**Result**: Minor version bump (features take precedence over fixes)

## 🎯 Commit Prefixes Reference

| Prefix | Type | Bump | Example |
|--------|------|------|---------|
| `BREAKING CHANGE:` | Breaking | Major | v1.0.0 → v2.0.0 |
| `major:` | Breaking | Major | v1.0.0 → v2.0.0 |
| `feat:` | Feature | Minor | v1.0.0 → v1.1.0 |
| `feature:` | Feature | Minor | v1.0.0 → v1.1.0 |
| `✨` | Feature | Minor | v1.0.0 → v1.1.0 |
| `fix:` | Fix | Patch | v1.0.0 → v1.0.1 |
| `bugfix:` | Fix | Patch | v1.0.0 → v1.0.1 |
| `🐛` | Fix | Patch | v1.0.0 → v1.0.1 |
| `patch:` | Fix | Patch | v1.0.0 → v1.0.1 |
| `chore:` | Maintenance | Patch | v1.0.0 → v1.0.1 |
| `docs:` | Documentation | Patch | v1.0.0 → v1.0.1 |
| `refactor:` | Refactor | Patch | v1.0.0 → v1.0.1 |
| `test:` | Tests | Patch | v1.0.0 → v1.0.1 |
| `perf:` | Performance | Patch | v1.0.0 → v1.0.1 |

## 🚫 What Won't Trigger a Release

Changes to these paths are ignored:
- `docs/**` - Documentation files
- `*.md` - Markdown files
- `.github/**` - Workflow files
- `examples/**` - Example scripts

Regular commits without the above prefixes won't trigger releases.

## 🔍 Checking Release Status

After pushing:

1. **Go to Actions tab** in GitHub
2. Look for **"Auto Release"** workflow
3. Check the summary:
   - ✅ Release created
   - ℹ️ No release needed

4. **If release was created:**
   - Check **"Create Release"** workflow for final release
   - View release at: `https://github.com/OWNER/REPO/releases`

## ⚙️ Manual Override

If you need to create a release manually:

### Option 1: Use the script
```bash
./create-release.sh
```

### Option 2: Manual tag
```bash
git tag -a v1.2.3 -m "Release v1.2.3"
git push origin v1.2.3
```

### Option 3: GitHub Actions UI
1. Go to Actions → Create Release
2. Run workflow manually
3. Enter version and type

## 📊 Version Strategy

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0 → 2.0.0): Breaking changes
- **MINOR** (1.0.0 → 1.1.0): New features, backward compatible
- **PATCH** (1.0.0 → 1.0.1): Bug fixes, backward compatible

## 💡 Best Practices

1. **Write clear commit messages**
   ```bash
   # Good
   feat: Add MongoDB authentication support
   fix: Resolve connection timeout in API loader
   
   # Bad
   update code
   fixes
   ```

2. **One logical change per commit**
   ```bash
   # Good
   git commit -m "feat: Add Redis loader"
   git commit -m "docs: Update Redis loader documentation"
   
   # Avoid
   git commit -m "feat: Add Redis, fix bugs, update docs"
   ```

3. **Use conventional commits**
   - Start with prefix
   - Clear description
   - Optional body with details

4. **Test before pushing**
   ```bash
   # Run tests locally
   pytest tests/
   
   # Check formatting
   black --check .
   flake8 .
   ```

## 🛠️ Troubleshooting

### Release not created?
- Check commit message has correct prefix
- Verify changes aren't in ignored paths
- Check Actions tab for errors

### Wrong version bump?
- Major > Minor > Patch in priority
- If you have mixed commits, highest priority wins
- Check the Auto Release workflow logs

### Tag already exists?
- Workflow will skip if tag exists
- Delete tag and re-push if needed:
  ```bash
  git tag -d v1.2.3
  git push origin :refs/tags/v1.2.3
  ```

## 📚 Additional Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

**Status**: ✅ Automatic releases enabled  
**Current Version**: Check VERSION file  
**Latest Release**: https://github.com/isathish/slm/releases
