# Git Hooks

This directory contains git hooks that are tracked in the repository. These hooks enforce code quality and consistency standards automatically.

## Quick Setup

**First time after cloning:**

```bash
./scripts/setup-hooks.sh
```

This one-time command configures git to use hooks from this directory instead of `.git/hooks/`.

**Verify setup:**
```bash
git config core.hooksPath
# Should output: /path/to/forgather/.githooks
```

## Available Hooks

### pre-commit

Automatically formats staged Python files before each commit.

**What it does:**
1. Finds all staged `.py` files
2. Runs `isort` to sort and organize imports
3. Runs `black` to format code style
4. Re-stages formatted files automatically
5. Shows which files were modified

**Example output:**
```
Running isort and black on staged Python files...
Running isort...
Running black...
Files were reformatted by isort and/or black:
  - src/forgather/ml/trainer/trainer.py
Re-staging modified files...
Formatting complete!
```

**Excluded files:**

Files matching patterns in `.formatting-ignore` are skipped. Common exclusions:
- Jinja2 templates with `.py` extensions (e.g., `modelsrc/templates/*.py`)
- Generated code
- Test fixtures requiring specific formatting

**Bypass when needed:**
```bash
git commit --no-verify -m "Skip formatting for this commit"
```

## Configuration

### Adding Exclusions

Edit `.formatting-ignore` at the repository root:

```bash
# Add patterns for files to exclude from formatting
echo "my_special_file.py" >> .formatting-ignore
echo "experiments/**/*.py" >> .formatting-ignore
```

Supports glob patterns (wildcards):
- `*.py` - All Python files in current directory
- `**/*.py` - All Python files recursively
- `modelsrc/templates/*.py` - Specific path pattern

### Requirements

The hooks require these tools:
- **isort** - `pip install isort`
- **black** - `pip install black`

These are included in Forgather's `setup.py` dependencies.

## Usage Examples

**Normal workflow:**
```bash
# Edit Python files
vim src/forgather/ml/trainer.py

# Stage and commit (formatting happens automatically)
git add src/forgather/ml/trainer.py
git commit -m "Update trainer"
```

**Adding new exclusions:**
```bash
# Exclude a template file
echo "modelsrc/new_template.py" >> .formatting-ignore
git add .formatting-ignore
git commit -m "Exclude new template from formatting"
```

**Bypassing for work-in-progress:**
```bash
# Commit without formatting (rare cases)
git commit --no-verify -m "WIP: Debugging syntax error"
```

## For Maintainers

### Modifying Hooks

1. Edit hook file: `vim .githooks/pre-commit`
2. Test locally by committing a Python file
3. Commit changes: `git add .githooks/pre-commit && git commit -m "Update hook"`
4. Developers get updates on next `git pull` (no reinstall needed)

### Adding New Hooks

1. Create hook file: `vim .githooks/new-hook`
2. Make executable: `chmod +x .githooks/new-hook`
3. Document in this README
4. Commit and push

### Hook Development Guidelines

- Use `#!/usr/bin/env bash` for portability
- Include `set -e` to exit on errors
- Provide clear user-facing output
- Handle empty file lists gracefully
- Exit with 0 for success, non-zero for failure
- Add comments for complex logic

## Troubleshooting

**Hook doesn't run:**
```bash
# Check configuration
git config core.hooksPath

# If not set, run setup script
./scripts/setup-hooks.sh
```

**isort/black not found:**
```bash
# Install formatting tools
pip install isort black

# Or reinstall Forgather
pip install -e .
```

**File not being formatted:**
- Check if file is staged: `git status`
- Check if file matches exclusion pattern in `.formatting-ignore`
- Try manual formatting: `isort file.py && black file.py`

**Hook fails with error:**
- Read error message carefully
- Check if Python file has syntax errors
- Try formatting manually to see detailed error
- Use `--no-verify` to bypass if needed

## Uninstalling

To stop using these hooks:

```bash
git config --unset core.hooksPath
```

To re-enable later:
```bash
./scripts/setup-hooks.sh
```

## See Also

- **Detailed documentation:** [docs/development/git-hooks.md](../docs/development/git-hooks.md)
- **Setup script:** [scripts/setup-hooks.sh](../scripts/setup-hooks.sh)
- **Exclusion patterns:** [.formatting-ignore](../.formatting-ignore)
- **Black formatter:** https://black.readthedocs.io/
- **isort:** https://pycqa.github.io/isort/
