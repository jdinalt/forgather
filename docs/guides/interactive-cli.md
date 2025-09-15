# Interactive CLI Guide

The Forgather Interactive CLI provides a powerful shell environment for working with Forgather projects, featuring advanced editor integration, multi-file selection, and smart completion.

## Getting Started

### Launching the Interactive CLI

```bash
# Start in current directory
forgather -i

# Start in specific project directory  
forgather -p /path/to/project -i
```

### Basic Commands

```bash
forgather> help              # Show all available commands
forgather> pwd               # Show current project directory
forgather> cd <directory>    # Change to project directory
forgather> configs           # List available templates
forgather> config <template> # Set current template
forgather> commands          # List available forgather commands
```

Note that tab-completion should work for most things.

## Editor Integration

The interactive CLI features editor integration with automatic detection and optimization for different editors.

We also have Forgather syntax highlighting plugins for [vim](../../syntax_highlighting/vim/vim-syntax-install.md) and [VS code](../../syntax_highlighting/vscode/README.md).

### Editor Selection Priority

The CLI automatically selects the best available editor in this order:

1. **VS Code Remote CLI** (when `VSCODE_IPC_HOOK_CLI` is set)
2. **Vim Clientserver** (when `VIM_SERVERNAME` is set and vim has `+clientserver`)
3. **User's `EDITOR` environment variable**
4. **Default vim**

### VS Code Integration

#### Automatic Detection
When running in a VS Code terminal (remote or local), the CLI automatically detects VS Code and opens files directly in the editor.

```bash
# VS Code terminal automatically sets this
echo $VSCODE_IPC_HOOK_CLI
# Output: /tmp/vscode-ipc-abc123-def456.sock
```

#### Manual Setup
You can copy the IPC socket from a VS Code session to use elsewhere:

```bash
# In VS Code terminal
echo $VSCODE_IPC_HOOK_CLI

# In other terminal/session
export VSCODE_IPC_HOOK_CLI=/tmp/vscode-ipc-abc123-def456.sock

# Now forgather will open files in VS Code
forgather interactive
forgather> edit
```

### Vim Clientserver Integration

#### Prerequisites
Vim must be compiled with `+clientserver` support:

```bash
vim --version | grep +clientserver
```

Most distributions don't seem to have this feature enabled. If you wish to use clientserver mode, you can build vim from source:

```bash
git clone https://github.com/vim/vim.git &&
cd vim &&
./configure &&
make -j 8

# You may wish to customize the install path. See "./configure --help"
sudo make install
```

#### Setup
1. **Start a vim server instance:**
   ```bash
   vim --servername myviminstance
   ```

2. **Set environment variable:**
   ```bash
   export VIM_SERVERNAME=myviminstance
   export EDITOR=/path/to/vim  # Optional: specify vim path
   ```

3. **Use in interactive CLI:**
   ```bash
   forgather interactive
   forgather> edit
   # Files will open as tabs in the existing vim server
   ```

#### Benefits
- **Persistent editing session** - Keep your vim configuration, plugins, and session
- **Fast file switching** - No startup time for new files
- **Unified workspace** - All template files in one vim instance
- **Plugin compatibility** - Works with vim plugins like NERDTree, fugitive, etc.

### Other Editors

#### Standard VS Code
```bash
export EDITOR=code
# Uses: code -r file1 file2  # Reuses window
```

## Multi-File Editing

The `edit` command supports sophisticated multi-file selection for efficient template editing.

### Selection Syntax

| Syntax | Description | Example |
|--------|-------------|---------|
| `5` | Single file | Selects file #5 |
| `1,3,7` | Multiple files | Selects files #1, #3, #7 |
| `1-5` | Range | Selects files #1 through #5 |
| `1-3,7,9-12` | Combined | Selects #1-3, #7, #9-12 |

### Interactive Selection Process

```bash
forgather> edit

Available templates:
==================================================

Project Configs:
   1. templates/configs/baseline.yaml
   2. templates/configs/experiment1.yaml
   3. templates/configs/experiment2.yaml

Base Templates:
   4. ../../templatelib/base/trainers/simple.yaml
   5. ../../templatelib/base/models/transformer.yaml

   0. Cancel
==================================================
You can select multiple files:
  - Single file: 5
  - Multiple files: 1,3,7
  - Ranges: 1-5,8,10-12

Select template(s) (0-3): 1,3
```

### Editor-Specific Multi-File Behavior

#### VS Code
- Opens all files as tabs in current window
- Preserves existing tabs
- Files appear in editor tab bar

#### Vim (Regular)
```bash
vim -p file1.yaml file2.yaml file3.yaml
```
- Opens files in vim tabs
- Use `gt`/`gT` to switch between tabs
- `:tabnew`, `:tabclose` for tab management

#### Vim (Clientserver)
```bash
vim --servername myserver --remote-tab file1.yaml
vim --servername myserver --remote-tab file2.yaml
```
- Adds files as new tabs to existing vim instance
- Preserves current session and configuration
- Immediate availability in existing vim session

## Help System

### Command-Specific Help

Each command supports standard `--help` flags:

```bash
forgather> edit --help
forgather> config --help
forgather> cd -h
```

### Built-in Help

```bash
forgather> help           # List all commands
forgather> help edit      # Help for specific command
```

### Environment Variables Summary

| Variable | Purpose | Example |
|----------|---------|---------|
| `VSCODE_IPC_HOOK_CLI` | VS Code IPC socket | `/tmp/vscode-ipc-*.sock` |
| `VIM_SERVERNAME` | Vim server instance name | `myviminstance` |
| `EDITOR` | Preferred editor | `vim`, `code`, `nvim` |

### Tab Completion

The CLI provides intelligent tab completion:

```bash
forgather> cd <TAB>         # Complete directory paths
forgather> config <TAB>     # Complete template names  
forgather> edit <TAB>       # Complete file paths
forgather> -t <TAB>         # Complete template names
```

### Command History

Commands are persistently stored and searchable:

- **Up/Down arrows** - Navigate command history
- **Ctrl+R** - Search command history
- **History file** - Stored in workspace root as `.forgather_history`

### Direct Template Usage

Set a current template for command shortcuts:

```bash
forgather> config baseline.yaml    # Set current template
forgather> pp                      # Uses baseline.yaml
forgather> train                   # Uses baseline.yaml
forgather> -t other.yaml pp        # Override for single command
```

### File Creation

The edit command can create new template files:

```bash
forgather> edit new_template.yaml
Template file doesn't exist. Create new_template.yaml? (y/N): y
# Creates file and opens in editor
```

## Troubleshooting

### VS Code Not Detected

**Problem**: Files open in vim instead of VS Code

**Solutions**:
1. Check IPC socket: `echo $VSCODE_IPC_HOOK_CLI`
2. Copy from VS Code terminal: `export VSCODE_IPC_HOOK_CLI=<value>`
3. Verify VS Code remote CLI exists

### Vim Clientserver Issues

**Problem**: `VIM_SERVERNAME` set but files open in new vim instance

**Solutions**:
1. Check clientserver support: `vim --version | grep clientserver`
2. Verify server is running: `vim --serverlist`
3. Start server: `vim --servername myinstance`

## Tips and Best Practices

### Efficient Workflows

1. **Set up vim clientserver** for persistent editing sessions
2. **Use ranges** for bulk template editing (`1-5,8-12`)
3. **Set current template** for quick command execution
4. **Enable VS Code IPC** in remote environments

### Template Organization

1. **Group related templates** for easier multi-selection
2. **Use descriptive names** for template identification
3. **Test configurations** with `pp` before training
4. **Version control** template changes

### Performance Tips

1. **Use tab completion** to avoid typing full paths
2. **Set EDITOR once** in your shell profile
3. **Keep vim server running** for instant file access
4. **Use command history** to repeat complex operations

---

*For more information on Forgather CLI commands, see the [CLI Reference](../reference/cli.md) or run `forgather --help`.*