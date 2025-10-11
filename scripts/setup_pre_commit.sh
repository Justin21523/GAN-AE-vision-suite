#!/bin/bash

# GAN-AE-VISION-SUITE Pre-commit Setup Script

set -e

echo "🔧 Setting up pre-commit hooks for GAN-AE-VISION-SUITE"

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "📥 Installing pre-commit..."
    pip install pre-commit
fi

# Install pre-commit hooks
echo "📝 Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Install development dependencies
echo "📦 Installing development dependencies..."
pip install -r requirements-dev.txt

# Run pre-commit on all files
echo "🔍 Running pre-commit on all files..."
pre-commit run --all-files

echo "✅ Pre-commit setup completed successfully!"
echo ""
echo "📋 Pre-commit hooks installed:"
echo "  - black (code formatting)"
echo "  - isort (import sorting)"
echo "  - ruff (linting)"
echo "  - mypy (type checking)"
echo "  - various pre-commit hooks"