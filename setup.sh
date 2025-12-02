#!/bin/bash

# SLM-Builder Quick Setup Script
# This script helps you get started with SLM-Builder

set -e

echo "🚀 SLM-Builder Quick Setup"
echo "=========================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

echo "📍 Checking Python version..."
python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" || {
    echo "❌ Python 3.8+ required. Found: $PYTHON_VERSION"
    exit 1
}
echo "✅ Python $PYTHON_VERSION detected"
echo ""

# Ask installation type
echo "Select installation type:"
echo "  1) CPU-only (minimal, for development)"
echo "  2) Full (with GPU support, recommended)"
echo "  3) Development (editable install with dev tools)"
echo ""
read -p "Enter choice [1-3]: " INSTALL_TYPE

case $INSTALL_TYPE in
    1)
        echo "📦 Installing CPU-only version..."
        pip install -e .
        ;;
    2)
        echo "📦 Installing full version with GPU support..."
        pip install -e ".[full]"
        ;;
    3)
        echo "📦 Installing development version..."
        pip install -e ".[full,dev]"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✅ Installation complete!"
echo ""

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "import slm_builder; print(f'   Version: {slm_builder.__version__}')" || {
    echo "❌ Verification failed"
    exit 1
}

# Check CLI
echo "🔍 Checking CLI..."
slm --version || {
    echo "❌ CLI verification failed"
    exit 1
}

echo ""
echo "✅ All checks passed!"
echo ""

# Next steps
echo "📚 Next Steps:"
echo ""
echo "1. Try the example:"
echo "   cd examples"
echo "   python build_qa_from_csv.py"
echo ""
echo "2. Build from your own data:"
echo "   slm build --source your_data.csv --task qa --recipe lora"
echo ""
echo "3. Launch annotation UI:"
echo "   slm annotate --source your_data.csv --task qa"
echo ""
echo "4. Read the documentation:"
echo "   cat README.md"
echo "   cat docs/INSTALLATION.md"
echo ""
echo "🎉 Happy building!"
