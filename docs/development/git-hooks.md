# Git Hooks

This document describes the git hooks setup for Forgather development.

## Overview

Forgather uses git hooks to automatically format Python code before commits. This ensures consistent code style across the project without manual intervention.

## Quick Start

```bash
# One-time setup after cloning the repository
./scripts/setup-hooks.sh
```

That's it! The hooks will now run automatically on every commit.

## What the Hooks Do

### Pre-commit Hook

The pre-commit hook runs before each commit and:

1. **Finds staged Python files** - Only processes files you're committing
2. **Applies formatting** - Runs `isort` and `black` on those files
3. **Re-stages changes** - Automatically adds formatted files back to the commit
4. **Respects exclusions** - Skips files matching patterns in `.formatting-ignore`

**Example workflow:**

```bash
# You edit some Python files with messy formatting
vim src/forgather/ml/trainer/trainer.py

# Add them to staging
git add src/forgather/ml/trainer/trainer.py

# Commit - hook runs automatically
git commit -m "Fix trainer bug"

# Output shows:
# Running isort and black on staged Python files...
# Running isort...
# Running black...
# Files were reformatted by isort and/or black:
#   - src/forgather/ml/trainer/trainer.py
# Re-staging modified files...
# Formatting complete!
```

## Configuration

### Formatting Exclusions (.formatting-ignore)

Some files should not be formatted (e.g., Jinja2 templates that look like Python). These are listed in `.formatting-ignore` at the repository root.

**Format:**
```
# Comments start with #
# One pattern per line
# Supports glob patterns

# Exclude specific file
modelsrc/templates/hf_causal.py

# Exclude directory pattern
modelsrc/templates/*.py
experimental/**/*.py

# Exclude by path component
**/test_fixtures/*.py
```

**Common exclusions:**
- Jinja2 templates with `.py` extension
- Generated code that shouldn't be reformatted
- Third-party code vendored into the repository
- Test fixtures that intentionally have specific formatting

### Git Configuration

The setup script configures git to use `.githooks/` instead of `.git/hooks/`:

```bash
git config core.hooksPath .githooks
```

This is a per-repository setting. Each developer runs the setup script once after cloning.

**To verify your configuration:**
```bash
git config core.hooksPath
# Should output: /path/to/forgather/.githooks
```

## Usage

### Normal Development

Just commit as usual:

```bash
git add my_changes.py
git commit -m "Add new feature"
# Hook runs automatically
```

### Bypassing the Hook

Sometimes you need to commit without formatting (rare, but possible):

```bash
git commit --no-verify -m "WIP: Testing unformatted code"
```

**When to bypass:**
- Committing deliberately unformatted code for testing
- Emergency hotfixes where formatting might introduce bugs
- Working with code that can't be formatted (syntax errors)

**Note:** Bypassing should be rare. CI/CD may reject unformatted code.

### Troubleshooting

**Hook doesn't run:**

1. Check configuration:
   ```bash
   git config core.hooksPath
   ```

2. If not set, run setup script:
   ```bash
   ./scripts/setup-hooks.sh
   ```

3. Verify hook is executable:
   ```bash
   ls -la .githooks/pre-commit
   # Should show -rwxr-xr-x (executable)
   ```

**Formatting fails:**

1. Check if `isort` and `black` are installed:
   ```bash
   which isort
   which black
   ```

2. Install if missing:
   ```bash
   pip install isort black
   ```

3. Test manually:
   ```bash
   isort your_file.py
   black your_file.py
   ```

**File should be excluded but isn't:**

1. Check `.formatting-ignore` syntax
2. Ensure pattern matches the file path (relative to repository root)
3. Test pattern matching:
   ```bash
   # In the hook script, patterns are matched with bash [[ ]]
   # Test if your pattern works:
   [[ "modelsrc/templates/test.py" == modelsrc/templates/*.py ]] && echo "Match!"
   ```

**Hook runs but doesn't format:**

Check if the file is staged:
```bash
git status
# Files must be in "Changes to be committed" section
```

## Updating Hooks

When hooks are updated in the repository:

```bash
git pull
# That's it! The updated hooks are now active
```

No reinstall needed - hooks are tracked in `.githooks/` and git is configured to use that directory.

## For Maintainers

### Modifying Hooks

1. Edit hooks in `.githooks/` directory:
   ```bash
   vim .githooks/pre-commit
   ```

2. Test changes locally:
   ```bash
   # Stage a Python file and commit to test
   git add test_file.py
   git commit -m "Test"
   ```

3. Commit hook changes:
   ```bash
   git add .githooks/pre-commit
   git commit -m "Update pre-commit hook to..."
   ```

4. Push changes:
   ```bash
   git push
   ```

All developers will get the updated hook on their next `git pull`.

### Adding New Hooks

1. Create new hook in `.githooks/`:
   ```bash
   vim .githooks/post-commit
   chmod +x .githooks/post-commit
   ```

2. Add documentation to `.githooks/README.md`

3. Update this document if needed

4. Commit and push:
   ```bash
   git add .githooks/post-commit .githooks/README.md
   git commit -m "Add post-commit hook for..."
   git push
   ```

### Hook Script Guidelines

When writing hooks:

- **Use bash** - More portable than shell-specific features
- **Set -e** - Exit on first error
- **Provide clear output** - Users should know what's happening
- **Handle edge cases** - Empty file lists, no Python files, etc.
- **Exit with correct codes** - 0 for success, non-zero for failure
- **Document behavior** - Add comments explaining complex logic

**Example hook structure:**

```bash
#!/usr/bin/env bash
# Description of what this hook does

set -e  # Exit on error

# Get relevant files
FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)

if [ -z "$FILES" ]; then
    # No Python files staged, exit successfully
    exit 0
fi

# Do the work
echo "Processing files..."
for file in $FILES; do
    # Process file
    process_file "$file"
done

echo "Hook complete!"
exit 0
```

## Alternative: Pre-commit Framework

For projects needing more sophisticated hook management, consider the [pre-commit framework](https://pre-commit.com/):

**Pros:**
- Large ecosystem of pre-built hooks
- Automatic updates via `pre-commit autoupdate`
- Standardized configuration
- Language-agnostic

**Cons:**
- Requires Python package installation
- Additional dependency to manage
- Less transparent than bash scripts

**Example config** is provided in `.pre-commit-config.yaml.example` for reference.

## See Also

- [.githooks/README.md](../../.githooks/README.md) - Quick reference for hooks
- [.formatting-ignore](../../.formatting-ignore) - Exclusion patterns
- [scripts/setup-hooks.sh](../../scripts/setup-hooks.sh) - Setup script
- [Black documentation](https://black.readthedocs.io/)
- [isort documentation](https://pycqa.github.io/isort/)
