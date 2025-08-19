# Running Multiple Forgather Experiments - Manual Guide

This document provides step-by-step instructions for running multiple ML experiments using the Forgather framework, particularly for users new to the system.

## Prerequisites

1. **Forgather Installation**: Ensure Forgather is installed and the `forgather` command is available in PATH
2. **Project Setup**: Navigate to a valid Forgather project directory with `meta.yaml` and `templates/` structure
3. **GPU Environment**: CUDA-capable GPUs with sufficient memory for your model

## Basic Single Experiment Workflow

### 1. Explore Available Configurations
```bash
# Navigate to your experiment project
cd examples/tiny_experiments/peak_memory

# List all available experiment configurations
forgather ls

# Get detailed project overview
forgather index
```

### 2. Validate Configuration Before Running
```bash
# Check preprocessed configuration (catches errors early)
forgather -t your_config.yaml pp

# View template inheritance chain
forgather -t your_config.yaml trefs

# List available build targets
forgather -t your_config.yaml targets
```

### 3. Run Single Experiment
```bash
# Basic training run
forgather -t your_config.yaml train

# With specific GPU selection
forgather -t your_config.yaml train -d 0

# With output logging
forgather -t your_config.yaml train 2>&1 | tee experiment_log.txt

# Dry run (show command without executing)
forgather -t your_config.yaml train --dry-run
```

### 4. Monitor Training Progress
```bash
# During training, you can monitor logs in real-time
tail -f output_models/*/runs/*/trainer_logs.json

# Check GPU utilization
nvidia-smi

# Monitor memory usage
watch -n 1 nvidia-smi
```

## Multi-GPU Manual Coordination

### Current System Limitations
- **No built-in job scheduling**: Each experiment must be manually started
- **No automatic GPU assignment**: Must manually specify GPUs to avoid conflicts
- **No resource management**: User must track which GPUs are available

### Manual Multi-GPU Strategy

#### 1. Plan Resource Allocation
```bash
# Check available GPUs
nvidia-smi

# List experiments to run
forgather ls

# Estimate resource requirements:
# - Each experiment typically needs 1 GPU
# - Peak memory requirements vary by configuration
# - Training time varies significantly by technique
```

#### 2. Run Experiments in Parallel

**Terminal 1 (GPU 0):**
```bash
cd examples/tiny_experiments/peak_memory
CUDA_VISIBLE_DEVICES=0 forgather -t control.yaml train 2>&1 | tee logs/control_gpu0.log
```

**Terminal 2 (GPU 1):**
```bash
cd examples/tiny_experiments/peak_memory  
CUDA_VISIBLE_DEVICES=1 forgather -t bfloat16.yaml train 2>&1 | tee logs/bfloat16_gpu1.log
```

**Terminal 3 (GPU 2):**
```bash
cd examples/tiny_experiments/peak_memory
CUDA_VISIBLE_DEVICES=2 forgather -t checkpoint.yaml train 2>&1 | tee logs/checkpoint_gpu2.log
```

**Continue for remaining GPUs...**

#### 3. Background Execution Alternative
```bash
# Create logs directory
mkdir -p logs

# Start background jobs with explicit GPU assignment
CUDA_VISIBLE_DEVICES=0 forgather -t control.yaml train > logs/control.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 forgather -t bfloat16.yaml train > logs/bfloat16.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 forgather -t checkpoint.yaml train > logs/checkpoint.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 forgather -t compile.yaml train > logs/compile.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 forgather -t fused.yaml train > logs/fused.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 forgather -t fused_checkpoint.yaml train > logs/fused_checkpoint.log 2>&1 &

# Monitor running jobs
jobs

# Check job completion
ps aux | grep forgather

# Monitor logs
tail -f logs/control.log
```

#### 4. Job Management
```bash
# Kill specific background job
kill %1  # kills job 1

# Kill all forgather processes (emergency stop)
pkill -f forgather

# Wait for all background jobs to complete
wait

# Check exit codes of completed jobs
echo "Control: $?" after each job completes
```

## Result Analysis Workflow

### 1. Collect Results
```bash
# Extract peak memory from all logs
grep "MAX CUDA MEMORY ALLOCATED" logs/*.log

# Extract training metrics
grep "train_runtime\|train_samples_per_second" logs/*.log

# View final evaluation metrics
grep "eval-loss" logs/*.log
```

### 2. Organize Results
```bash
# Create summary table
echo "Experiment,Peak_Memory_GiB,Training_Speed_samples_per_sec" > results_summary.csv

# Parse logs and populate table (manual process)
# Extract values from each log file and format as CSV
```

## Best Practices

### Resource Management
1. **Always specify GPU explicitly** using `CUDA_VISIBLE_DEVICES` or `-d` flag
2. **Monitor GPU memory** before starting new experiments
3. **Stagger job starts** to avoid initialization conflicts
4. **Use consistent output directories** to avoid conflicts

### Logging and Debugging
1. **Always capture logs** for later analysis
2. **Use descriptive log filenames** including configuration name
3. **Check configuration preprocessing** before training (`forgather -t config.yaml pp`)
4. **Validate all configs** before starting batch runs (`forgather ls`)

### Error Recovery
1. **Check logs immediately** if training fails to start
2. **Verify GPU availability** with `nvidia-smi`
3. **Ensure no conflicting processes** using same GPU
4. **Restart individual failed experiments** rather than entire batch

## Common Issues and Solutions

### GPU Conflicts
**Problem**: Multiple processes trying to use same GPU
**Solution**: Always use `CUDA_VISIBLE_DEVICES=N` for explicit assignment

### Memory Errors
**Problem**: GPU out of memory
**Solution**: Check `nvidia-smi`, reduce batch size, or use memory optimization techniques

### Configuration Errors
**Problem**: Training fails to start due to config issues
**Solution**: Run `forgather -t config.yaml pp` to debug preprocessing

### Job Tracking
**Problem**: Lost track of which experiments are running/completed
**Solution**: Use consistent naming scheme and job monitoring commands

## Next Steps for Automation

The manual process reveals several automation opportunities:
1. **Automatic GPU assignment** based on availability
2. **Job queue management** with priority scheduling  
3. **Resource monitoring** and dynamic allocation
4. **Automatic result collection** and comparison
5. **Failure handling** and automatic restart
6. **Configuration validation** before job submission

This manual process provides the foundation for designing an automated experiment management system.