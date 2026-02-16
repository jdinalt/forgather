#!/usr/bin/env bash
# Setup script to configure git hooks for this repository

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.githooks"

echo "Setting up git hooks..."
echo ""

# Configure git to use .githooks directory
git config core.hooksPath "$HOOKS_DIR"

echo "✓ Configured git to use hooks from .githooks/"
echo ""
echo "Pre-commit hook installed!"
echo "  - Automatically runs isort and black on staged Python files"
echo "  - Respects exclusions in .formatting-ignore"
echo "  - Bypass with: git commit --no-verify"
echo ""
