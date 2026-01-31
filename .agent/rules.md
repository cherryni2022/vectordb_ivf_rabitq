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

## Testing Rule

**Every new feature must include unit tests and benchmark comparisons.**

When implementing a new feature:

1. Create unit tests in `tests/test_<feature>.py`
2. Test all core functionality (initialization, build, search, persistence)
3. Create or update benchmark scripts in `benchmarks/`
4. Compare performance with existing implementations

### Example

```bash
# After adding a new index type
tests/test_hnswlib.py        # Unit tests
benchmarks/benchmark_ivf_vs_hnsw.py  # Performance comparison
```

## Dependency Management Rule

**Third-party dependencies must be added to pyproject.toml.**

When adding a new library:

1. Add required dependencies to `[project.dependencies]` or appropriate optional group
2. Use version constraints (e.g., `>=1.0.0`)
3. Consider adding to optional groups if not always needed

### Dependency Groups

| Group | Purpose |
|-------|---------|
| `dependencies` | Core required packages |
| `dev` | Development/testing tools |
| `jit` | JIT acceleration (numba) |
| `gpu` | GPU acceleration (torch) |
| `accel` | All acceleration features |

### Example

```toml
# Adding a new optional dependency
[project.optional-dependencies]
hnsw = [
    "hnswlib>=0.7.0",
]
```
