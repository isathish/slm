# 🚀 Quick Reference - Documentation & Releases

## 📚 Documentation Location

All documentation is in `docs/` directory:
- **Index**: [docs/README.md](docs/README.md)
- **Wiki**: https://github.com/isathish/slm/wiki (auto-published)

## 🔖 Creating a Release

### Quick Release (Recommended)
1. Go to **GitHub Actions** → **"Version Bump"**
2. Click **"Run workflow"**
3. Select type: `major` | `minor` | `patch`
4. Enable **"Create release"**
5. Click **"Run workflow"**

### Manual Release
```bash
git tag -a v1.0.0 -m "Release 1.0.0"
git push origin v1.0.0
# Release workflow auto-triggers
```

## 📊 Version Types

| Type | Example | Use Case |
|------|---------|----------|
| **major** | 1.0.0 → 2.0.0 | Breaking changes, major features |
| **minor** | 1.0.0 → 1.1.0 | New features, backward compatible |
| **patch** | 1.0.0 → 1.0.1 | Bug fixes, minor improvements |

## 🔄 Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **publish-wiki.yml** | Push to `main` (docs/ changes) | Publish docs to Wiki |
| **release.yml** | Push version tag (`v*.*.*`) | Create GitHub release |
| **version-bump.yml** | Manual dispatch | Bump version & tag |

## 📝 Before Release Checklist

```bash
# 1. Update CHANGELOG.md
# 2. Run tests
pytest tests/

# 3. Check formatting
black --check slm_builder/ examples/
isort --check-only slm_builder/ examples/
flake8 slm_builder/ examples/ --max-line-length=100

# 4. Verify version
cat VERSION
grep __version__ slm_builder/__init__.py

# 5. Create release (see above)
```

## 🌐 Access Points

- **Docs**: `docs/README.md`
- **Wiki**: `https://github.com/isathish/slm/wiki`
- **Releases**: `https://github.com/isathish/slm/releases`
- **Actions**: `https://github.com/isathish/slm/actions`

## 📖 Documentation Publishing

**Automatic**: Pushes to `main` with `docs/` changes

**Manual**:
- Actions → "Publish Documentation to Wiki" → Run workflow

## 🎯 Current Status

**Version**: 1.0.0 (Initial Release)  
**Status**: ✅ Ready for release

---

For detailed guide, see [SETUP_COMPLETE.md](SETUP_COMPLETE.md)
