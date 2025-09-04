# VSCode Syntax Highlighting for Forgather

This directory contains a Visual Studio Code extension for syntax highlighting of Forgather configuration files.

## Installation

### For Remote Development (Docker, Codespaces, Remote Containers)

#### Option 1: Server Extensions Directory (Recommended)
1. Copy the extension to the server's extensions directory:
   ```bash
   mkdir -p ~/.vscode-server/extensions/forgather-syntax-1.0.0
   cp -r syntax_highlighting/vscode/* ~/.vscode-server/extensions/forgather-syntax-1.0.0/
   ```

2. **Important**: A full server restart or window reload may be required. Sometimes the extension won't appear immediately and may require disconnecting/reconnecting to the remote environment.

#### Option 2: Create VSIX and Install via Command Line
1. Create a VSIX file using the Docker method (see `package-manually.md`)
2. Install via command line:
   ```bash
   code-server --install-extension forgather-syntax-1.0.0.vsix
   ```

### For Local Development

#### Option 1: Manual Installation
1. Copy this entire `vscode` directory to your VSCode extensions folder:

   **Windows:**
   ```cmd
   copy vscode %USERPROFILE%\.vscode\extensions\forgather-syntax-1.0.0
   ```

   **macOS:**
   ```bash
   cp -r vscode ~/.vscode/extensions/forgather-syntax-1.0.0
   ```

   **Linux:**
   ```bash
   cp -r vscode ~/.vscode/extensions/forgather-syntax-1.0.0
   ```

2. Restart VSCode or run "Developer: Reload Window" from the command palette

### Option 2: Package as VSIX (For Distribution)

If you want to create a proper extension package:

1. Install the VSCode Extension Manager (Note: there may be Node.js version compatibility issues):
   ```bash
   npm install -g @vscode/vsce
   ```

2. From the `vscode` directory, package the extension:
   ```bash
   cd vscode
   vsce package
   ```

3. Install the generated `.vsix` file:
   ```bash
   code --install-extension forgather-syntax-1.0.0.vsix
   ```

**Troubleshooting VSCE Issues:**
If you encounter Node.js compatibility errors with `vsce`, try:
- Using the newer `@vscode/vsce` package instead of `vsce`
- Using a different Node.js version (try Node 16 or 20)
- Or simply use the manual installation method (Option 1)

## File Association

After installation, you have several options for associating files with the Forgather syntax:

### Option 1: Modify VSCode Settings

Add to your VSCode `settings.json`:

```json
{
  "files.associations": {
    "**/forgather/**/*.yaml": "forgather-config",
    "**/templates/**/*.yaml": "forgather-config"
  }
}
```

### Option 2: Manual Language Selection

1. Open a Forgather configuration file
2. Click the language indicator in the bottom-right corner of VSCode
3. Select "Forgather Configuration" from the language list (search for "forgather" if needed)

### Option 3: Workspace Settings (Recommended for Remote Development)

For specific projects, create or edit `.vscode/settings.json` in your project root:

```json
{
  "files.associations": {
    "**/*.yaml": "forgather-config"
  }
}
```

This is particularly useful for remote development environments where you want the settings to be part of the project and shared across different development setups.

### Quick Installation Script

For remote environments, you can use the provided installation script:

```bash
cd syntax_highlighting/vscode
./install-remote.sh
```

This script will automatically detect your environment and install to the appropriate directory.

## Features

The extension provides syntax highlighting for:

- **Jinja2 line statements**: `--`, `<<`, `>>`, `==`, `=>`
- **Jinja2 standard syntax**: `{{ }}`, `{% %}`, `{# #}`
- **Line comments**: `##`
- **Custom YAML tags**: `!call`, `!singleton`, `!factory`, `!partial`, `!var`, etc.
- **Template split markers**: `#---- template.name ----`
- **YAML anchors and aliases**: `&name`, `*name`
- **Import specifications**: Module paths in tags
- **Named nodes**: `@name` suffixes
- **Dot-name elision**: `.define` keys
- **Jinja2 keywords, variables, strings, filters, and functions**

## Color Themes

The syntax highlighting uses standard TextMate scopes, so it will work with any VSCode color theme. The main scopes used are:

- `comment.line` and `comment.block` - Comments
- `punctuation.definition.template` - Jinja2 delimiters
- `keyword.control` - Jinja2 keywords
- `variable.other` - Variables
- `string.quoted` - Strings
- `storage.type.tag` - YAML tags
- `entity.name.tag` - Named nodes
- `support.function` - Functions and filters

## Testing

After installation:

1. Open any `.yaml` file in a Forgather project
2. Select "Forgather Configuration" as the language (if not auto-detected)
3. Verify that custom syntax elements are properly highlighted

**Note**: The language appears as "Forgather Configuration" in the language picker, but the language ID for settings is `forgather-config`.

## Troubleshooting

- **Extension not appearing**: 
  - Make sure the directory name matches `forgather-syntax-1.0.0`
  - For remote development, may require disconnecting/reconnecting or restarting the remote server
  - Check that the extension appears in: `code --list-extensions | grep forgather`
- **No syntax highlighting**: 
  - Check that the language is set to "Forgather Configuration" in the bottom-right corner
  - Verify file associations use `forgather-config` (not `forgather`)
- **Files not auto-detected**: 
  - Ensure `.vscode/settings.json` uses the correct language ID: `forgather-config`
  - Try more specific patterns like `**/templates/**/*.yaml` instead of `**/*.yaml`
- **Remote development issues**:
  - Extension may not load immediately after installation
  - Try creating a test file and manually selecting the language first
  - Ensure VSCode CLI environment is set up: `export VSCODE_IPC_HOOK_CLI=...`

## VSCode CLI Setup for Remote Development

To use `code` commands in remote environments, set up the environment:

```bash
export VSCODE_IPC_HOOK_CLI=/tmp/vscode-ipc-[your-session-id].sock
alias code=/home/[user]/.vscode-server/bin/[version]/bin/remote-cli/code
```

Then verify the extension is installed:
```bash
code --list-extensions | grep forgather
# Should output: forgather.forgather-syntax
```