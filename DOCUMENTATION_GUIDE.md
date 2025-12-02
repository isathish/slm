# 📖 How to Use Documentation & Release System

## 🎉 What's Been Set Up

All markdown documentation has been moved to `docs/` directory and automated publishing has been configured for:

1. **GitHub Pages** - Beautiful documentation website
2. **GitHub Wiki** - Searchable wiki pages
3. **Automated Releases** - Semantic versioning with one click

---

## 📚 Documentation Publishing

### GitHub Pages (Automatic)

Your documentation is automatically published to: **https://isathish.github.io/slm/**

**How it works:**
- Every push to `main` branch that changes `docs/**` files
- GitHub Actions builds with Jekyll
- Deploys to GitHub Pages
- Usually takes 2-3 minutes

**Workflow file:** `.github/workflows/jekyll-gh-pages.yml`

### GitHub Wiki (Automatic)

Your documentation is synced to: **https://github.com/isathish/slm/wiki**

**How it works:**
- Every push to `main` branch that changes `docs/**` files
- Automatically copies markdown files to wiki
- Creates sidebar navigation
- Updates within 1-2 minutes

**Workflow file:** `.github/workflows/publish-wiki.yml`

### Enabling GitHub Pages

1. Go to repository **Settings** → **Pages**
2. Under "Build and deployment":
   - Source: `GitHub Actions`
3. Save and wait for deployment

---

## 🔖 Creating Releases

### Option 1: Using GitHub Actions UI (Recommended)

1. Go to **Actions** tab in your repository
2. Click **"Version and Release"** workflow
3. Click **"Run workflow"** button
4. Select version type:
   - **major** - Breaking changes (1.0.0 → 2.0.0)
   - **minor** - New features (1.0.0 → 1.1.0)
   - **patch** - Bug fixes (1.0.0 → 1.0.1)
5. Check "Pre-release" if needed
6. Click **"Run workflow"**

**What happens automatically:**
- ✅ Version bumped in all files
- ✅ Git tag created (e.g., v1.1.0)
- ✅ Changelog generated from commits
- ✅ GitHub Release created
- ✅ All changes committed and pushed

### Option 2: Manual Bump (Advanced)

```bash
# Install bump2version
pip install bump2version

# Bump version
bump2version patch  # or minor, or major

# Push with tags
git push origin main --tags
```

---

## 📝 Version Management

### Current Version: 1.0.0

Version is tracked in:
- `VERSION` file
- `pyproject.toml`
- `setup.py`
- `slm_builder/__init__.py`
- `docs/README.md`

All updated automatically by bump2version!

### Semantic Versioning

We follow [SemVer](https://semver.org/):

- **MAJOR** (1.0.0 → 2.0.0): Breaking changes
  - Changed API
  - Removed features
  - Incompatible updates

- **MINOR** (1.0.0 → 1.1.0): New features
  - New functionality
  - Backward compatible
  - New data sources

- **PATCH** (1.0.0 → 1.0.1): Bug fixes
  - Bug fixes
  - Performance improvements
  - Documentation updates

---

## 📋 Commit Message Convention

For better changelogs, use prefixes:

- `feat:` or `✨` - New features
- `fix:` or `🐛` - Bug fixes
- `docs:` or `📚` - Documentation
- `chore:` or `🔧` - Maintenance

**Examples:**
```bash
git commit -m "feat: Add MongoDB data loader"
git commit -m "fix: Resolve database connection timeout"
git commit -m "docs: Update installation guide"
git commit -m "chore: Update dependencies"
```

These prefixes are automatically categorized in release notes!

---

## 🔄 Updating Documentation

### Making Changes

1. Edit files in `docs/` directory
2. Commit and push to `main` branch
3. GitHub Actions automatically:
   - Publishes to GitHub Pages
   - Syncs to Wiki

### Adding New Documentation

1. Create new `.md` file in `docs/`
2. Add to `docs/_config.yml` nav_order (optional)
3. Link from `docs/index.md`
4. Push changes

---

## 🚀 Quick Release Example

### Scenario: You fixed a bug

1. Make your code changes
2. Commit: `git commit -m "fix: Resolve data loading issue"`
3. Push: `git push origin main`
4. Go to Actions → "Version and Release"
5. Run workflow → Select **patch**
6. Wait 2-3 minutes
7. Check: New release v1.0.1 created!

### Scenario: You added a feature

1. Make your feature changes
2. Commit: `git commit -m "feat: Add Redis data loader"`
3. Push: `git push origin main`
4. Go to Actions → "Version and Release"
5. Run workflow → Select **minor**
6. Wait 2-3 minutes
7. Check: New release v1.1.0 created!

---

## 📊 Checking Status

### GitHub Pages Status
- Go to **Settings** → **Pages**
- See deployment status and URL

### GitHub Actions Status
- Go to **Actions** tab
- See all workflow runs
- Check logs if something fails

### GitHub Releases
- Go to **Releases** (sidebar)
- See all published releases
- Download assets

---

## 🛠️ Troubleshooting

### GitHub Pages not updating?

1. Check **Actions** tab for errors
2. Verify **Settings** → **Pages** is enabled
3. Make sure `docs/_config.yml` has correct `baseurl`

### Wiki not syncing?

1. Check if wiki is enabled: **Settings** → **Features** → "Wikis"
2. Check **Actions** tab for errors
3. Manually create wiki first: Go to **Wiki** tab → "Create the first page"

### Release workflow failing?

1. Check **Actions** tab for error logs
2. Verify all version files exist:
   - `VERSION`
   - `pyproject.toml`
   - `setup.py`
   - `slm_builder/__init__.py`
3. Make sure `.bumpversion.cfg` is present

---

## 📁 File Structure

```
slm/
├── .github/
│   └── workflows/
│       ├── jekyll-gh-pages.yml    # GitHub Pages
│       ├── publish-wiki.yml       # Wiki sync
│       └── release.yml            # Releases
├── docs/
│   ├── _config.yml                # Jekyll config
│   ├── index.md                   # Homepage
│   ├── README.md                  # Main guide
│   └── [all other .md files]
├── README.md                      # Root README
├── VERSION                        # Version number
├── .bumpversion.cfg               # Bump config
└── [rest of project files]
```

---

## ✅ Checklist for First Release

- [x] All docs moved to `docs/`
- [x] Jekyll configured
- [x] GitHub Pages workflow ready
- [x] Wiki publishing workflow ready
- [x] Release workflow configured
- [x] Version files in place
- [x] `.bumpversion.cfg` created
- [ ] Enable GitHub Pages (Settings → Pages)
- [ ] Enable Wiki (Settings → Features)
- [ ] Create first release (Actions → Version and Release)

---

## 🎓 Next Steps

1. **Enable GitHub Pages**:
   - Settings → Pages → Source: GitHub Actions

2. **Enable Wiki**:
   - Settings → Features → Check "Wikis"
   - Visit Wiki tab and create first page

3. **Create First Release**:
   - Actions → Version and Release
   - Run workflow → patch
   - Check release at: github.com/isathish/slm/releases

4. **Test Documentation**:
   - Visit: https://isathish.github.io/slm/
   - Visit: https://github.com/isathish/slm/wiki

---

## 📞 Support

If you encounter issues:

1. Check **Actions** tab for error logs
2. Review workflow files in `.github/workflows/`
3. Verify all setup steps completed
4. Check GitHub documentation

---

**Setup Date**: December 2, 2025  
**Version**: 1.0.0  
**Status**: ✅ Ready to use!

🎉 **Happy Documenting and Releasing!**
