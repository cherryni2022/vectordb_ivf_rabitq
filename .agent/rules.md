# Agent Rules

These rules guide agent behavior when working on this project.

## Git Commit Rule

**Every code or documentation change must be committed to git.**

After making any changes to:
- Source code files (`.py`, `.js`, `.ts`, etc.)
- Documentation files (`.md`, `.rst`, `.txt`)
- Configuration files (`pyproject.toml`, `requirements.txt`, etc.)
- Test files

The agent should:

1. Stage the changed files:
   ```bash
   git add <changed-files>
   ```

2. Commit with a descriptive message:
   ```bash
   git commit -m "<type>: <brief description>"
   ```

### Commit Message Format

Use conventional commit format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

### Examples

```bash
# After updating documentation
git add README.md CLAUDE.md
git commit -m "docs: update environment setup instructions with uv"

# After fixing a bug
git add vectordb/core/vector_db.py
git commit -m "fix: handle empty cluster edge case in IVF search"

# After adding a new feature
git add vectordb/quantization/true_rabitq.py tests/test_true_rabitq.py
git commit -m "feat: add True RaBitQ quantization implementation"
```
