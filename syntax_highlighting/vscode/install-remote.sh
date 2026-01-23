#!/bin/bash

# Installation script for VSCode remote development environments
# (Docker containers, Codespaces, Remote-SSH, etc.)

set -e

EXTENSION_ID="forgather-syntax-1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect VSCode server type and set appropriate extension directory
if [ -n "$VSCODE_IPC_HOOK_CLI" ] || [ -n "$VSCODE_SERVER_PORT" ]; then
    # Remote development environment detected
    EXTENSIONS_DIR="$HOME/.vscode-server/extensions"
    echo "🔍 Remote VSCode environment detected"
elif command -v code-server &> /dev/null; then
    # code-server environment
    EXTENSIONS_DIR="$HOME/.local/share/code-server/extensions"
    echo "🔍 code-server environment detected"
else
    # Fallback to local VSCode
    EXTENSIONS_DIR="$HOME/.vscode/extensions"
    echo "🔍 Local VSCode environment assumed"
fi

INSTALL_DIR="$EXTENSIONS_DIR/$EXTENSION_ID"

echo "📁 Installing Forgather syntax extension to: $INSTALL_DIR"

# Create extension directory
mkdir -p "$INSTALL_DIR"

# Copy all files except this script
find "$SCRIPT_DIR" -maxdepth 1 -type f ! -name "install-remote.sh" ! -name "*.md" -exec cp {} "$INSTALL_DIR/" \;
cp -r "$SCRIPT_DIR/syntaxes" "$INSTALL_DIR/"

echo "✅ Extension installed successfully!"
echo ""
echo "Next steps:"
echo "1. Reload your VSCode window (Ctrl+Shift+P → 'Developer: Reload Window')"
echo "2. Open a .yaml file in your Forgather project"
echo "3. Set the language to 'Forgather Configuration' using the language selector in the bottom-right"
echo "4. Or add file associations to your VSCode settings:"
echo ""
echo "   \"files.associations\": {"
echo "     \"**/forgather/**/*.yaml\": \"forgather-config\","
echo "     \"**/templates/**/*.yaml\": \"forgather-config\""
echo "   }"
echo ""
echo "Note: The language appears as 'Forgather Configuration' but the language ID is 'forgather-config'"
echo ""
echo "🎨 Enjoy syntax highlighting for your Forgather configurations!"