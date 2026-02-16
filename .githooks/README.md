# Git Hooks

This directory contains git hooks that are tracked in the repository.

## Setup

To enable these hooks, run:

```bash
./scripts/setup-hooks.sh
```

This configures git to use hooks from this directory instead of `.git/hooks/`.

## Available Hooks

### pre-commit

Automatically formats staged Python files using:
- **isort**: Sorts and organizes imports
- **black**: Formats code to consistent style

**Excluded files**: Files matching patterns in `.formatting-ignore` are skipped (e.g., Jinja2 templates).

**Bypass hook**: Use `git commit --no-verify` to skip formatting checks.

## Adding/Modifying Hooks

1. Edit hooks in this directory (`.githooks/`)
2. Make them executable: `chmod +x .githooks/your-hook`
3. Commit changes
4. Other developers will get updates automatically after running `git pull`

## Uninstalling

To stop using these hooks:

```bash
git config --unset core.hooksPath
```
