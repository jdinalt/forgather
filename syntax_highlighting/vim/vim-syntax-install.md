# Vim Syntax Highlighting for Forgather

This directory contains a vim syntax file (`forgather.vim`) for proper syntax highlighting of Forgather configuration files.

## Installation

### Option 1: Manual Installation

1. Copy `forgather.vim` to your vim syntax directory:
   ```bash
   mkdir -p ~/.vim/syntax/
   cp forgather.vim ~/.vim/syntax/
   ```

2. Add filetype detection to your `~/.vim/filetype.vim` (create if it doesn't exist):
   ```vim
   augroup filetypedetect
     au! BufRead,BufNewFile *.yaml setfiletype forgather
   augroup END
   ```

### Option 2: Alternative Filetype Detection

If you only want syntax highlighting for specific directories or patterns, you can be more selective:

```vim
augroup filetypedetect
  " Only apply to files in forgather projects
  au! BufRead,BufNewFile */forgather/*/*.yaml setfiletype forgather
  au! BufRead,BufNewFile */templates/*/*.yaml setfiletype forgather
augroup END
```

### Option 3: Using Plugin Manager

If you use a plugin manager like vim-plug, Vundle, or Pathogen, you can create a simple plugin structure:

1. Create the directory structure in your plugin directory
2. Place the syntax file and add ftdetect rules

## Features

The syntax file provides highlighting for:

- **Jinja2 line statements**: `--`, `<<`, `>>`, `==`, `=>`
- **Line comments**: `##`  
- **Custom YAML tags**: `!call`, `!singleton`, `!factory`, `!partial`, `!var`, etc.
- **Template split markers**: `#---- template.name ----`
- **YAML anchors and aliases**: `&name`, `*name`
- **Import specifications**: Module paths in tags
- **Named nodes**: `@name` suffixes
- **Dot-name elision**: `.define` keys
- **Jinja2 keywords and expressions**

## Testing

After installation, open any `.yaml` file in your forgather project and you should see proper syntax highlighting for all the custom syntax elements.