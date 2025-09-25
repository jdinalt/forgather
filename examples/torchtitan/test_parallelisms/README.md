# Compare Parallelisms

The "control" configuration is a fairly simple configuration, with one GPU, and 10 gradient accumulation steps, over 500 steps.

The others use various parallel strateges, but the same effective batch size (320). In theory, this should produce reasonably comparable results.

## Running the Tests

```bash
# List configurations
forgather ls

# Run a test configuration
forgather -t TEST_NAME train

# Monitor with Tenstor Board
forgather tb

# Run interactive shell, with config tools
forgather -i
```
