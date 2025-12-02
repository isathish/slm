#!/bin/bash
# Script to create a new release for SLM Builder

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     SLM Builder Release Creator         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Not in a git repository${NC}"
    exit 1
fi

# Check if there are uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}⚠️  Warning: You have uncommitted changes${NC}"
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get current version from VERSION file
if [ -f VERSION ]; then
    CURRENT_VERSION=$(cat VERSION)
    echo -e "${GREEN}Current version: ${CURRENT_VERSION}${NC}"
else
    echo -e "${YELLOW}⚠️  VERSION file not found. Starting from 1.0.0${NC}"
    CURRENT_VERSION="1.0.0"
fi

# Parse current version
IFS='.' read -r -a VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR="${VERSION_PARTS[0]}"
MINOR="${VERSION_PARTS[1]}"
PATCH="${VERSION_PARTS[2]}"

echo ""
echo "Select release type:"
echo "  1) Major release (${MAJOR}.${MINOR}.${PATCH} → $((MAJOR+1)).0.0) - Breaking changes"
echo "  2) Minor release (${MAJOR}.${MINOR}.${PATCH} → ${MAJOR}.$((MINOR+1)).0) - New features"
echo "  3) Patch release (${MAJOR}.${MINOR}.${PATCH} → ${MAJOR}.${MINOR}.$((PATCH+1))) - Bug fixes"
echo "  4) Custom version"
echo "  0) Cancel"
echo ""
read -p "Enter choice [1-4, 0]: " choice

case $choice in
    1)
        NEW_VERSION="$((MAJOR+1)).0.0"
        RELEASE_TYPE="major"
        ;;
    2)
        NEW_VERSION="${MAJOR}.$((MINOR+1)).0"
        RELEASE_TYPE="minor"
        ;;
    3)
        NEW_VERSION="${MAJOR}.${MINOR}.$((PATCH+1))"
        RELEASE_TYPE="patch"
        ;;
    4)
        read -p "Enter new version (x.y.z): " NEW_VERSION
        if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo -e "${RED}❌ Error: Invalid version format${NC}"
            exit 1
        fi
        RELEASE_TYPE="custom"
        ;;
    0)
        echo "Cancelled"
        exit 0
        ;;
    *)
        echo -e "${RED}❌ Error: Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${YELLOW}Creating release v${NEW_VERSION}${NC}"
echo ""

# Confirm
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# Update VERSION file
echo "$NEW_VERSION" > VERSION
echo -e "${GREEN}✓ Updated VERSION file${NC}"

# Commit version change
git add VERSION
git commit -m "🔖 Bump version to v${NEW_VERSION}" || true

# Create and push tag
git tag -a "v${NEW_VERSION}" -m "Release v${NEW_VERSION}"
echo -e "${GREEN}✓ Created tag v${NEW_VERSION}${NC}"

# Push changes
echo ""
echo "Pushing changes to remote..."
git push origin main
git push origin "v${NEW_VERSION}"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Release Created! 🎉              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo "Version: v${NEW_VERSION}"
echo "Type: ${RELEASE_TYPE}"
echo ""
echo "📋 Next steps:"
echo "  1. Check GitHub Actions for release workflow"
echo "  2. Release will be published automatically"
echo "  3. View at: https://github.com/$(git config --get remote.origin.url | sed 's/.*://;s/.git$//')/releases"
echo ""
