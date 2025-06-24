# Lessons Learned: External Workspace Development

## Notes to Future Self

This document captures the key lessons learned while creating the ALiBi GLU external workspace example. Use this as a reference to avoid repeating the same mistakes and to streamline future project development.

## Critical Setup Issues & Solutions

### 1. External Workspace Configuration
**Problem**: Initial confusion about workspace structure and base_directories.yaml
**Solution**: The external workspace requires a specific structure:
```
workspace_root/
├── forgather_workspace/
│   └── base_directories.yaml  # CRITICAL: Points to Forgather installation
└── project_name/
    ├── meta.yaml
    ├── templates/
    └── output_models/
```

**Key Point**: `base_directories.yaml` must set `ns.forgather_dir` to the absolute path of your Forgather installation. This is how external workspaces find the templatelib.

### 2. Template Inheritance Complexity
**Problem**: Jinja2 template inheritance doesn't work the same way as Python class inheritance
**Solution**: Use split document approach with `#--------------------` syntax for complex inheritance

**Wrong Way** (Jinja2 inheritance limitations):
```yaml
-- extends 'project.yaml'
-- block trainer_definition
    -- include 'baseline.trainer_config'  # This creates circular dependencies
-- endblock
```

**Right Way** (Split documents):
```yaml
-- extends 'project.yaml'

-- block trainer_definition
    -- include 'baseline.trainer_config'
-- endblock

#-------------------- baseline.trainer_config --------------------
-- extends 'project.trainer_config'

-- block trainer_args
    == super()
    learning_rate: 3e-4
-- endblock
```

### 3. Memory Management & OOM Issues
**Problem**: Training failed with OOM on variable-length sequences
**Solution**: Add truncation blocks to configurations:

```yaml
-- block datacollator
    == super()
    truncation: True
    max_length: 512
-- endblock

-- block tokenize_args
    == super()
    max_length: 512
-- endblock
```

**Lesson**: Always start with 512-token truncation for initial experiments, then increase if needed.

### 4. CLI Usage & Environment Variables
**Problem**: Confused about RANK environment variable requirement
**Solution**: 
- For single GPU: `fgcli.py -t config.yaml train -d 0`
- For multi-GPU: Use torchrun or accelerate launch
- RANK=0 was a workaround; the training script should be fixed to assume single process when RANK is unset

### 5. Architecture Compatibility Issues
**Problem**: Tried to use ALiBi + Absolute PE together, which is contradictory
**Solution**: 
- ALiBi models should use `NullPE` (no positional encoding)
- Absolute PE models should use standard attention
- Never mix relative and absolute positional encoding

### 6. Project Constructor Simplification
**Problem**: Used explicit default parameters in notebook
**Solution**: Just use `Project()` - it automatically finds the workspace configuration
```python
# Wrong
proj = Project(project_dir=".", config_name="")

# Right  
proj = Project()
```

### 7. Template Search and Component Location
**Problem**: Difficulty finding existing components and understanding template structure
**Solution**: Use systematic approach:
1. Search for similar existing templates first
2. Check `/modelsrc/transformer/` for component implementations
3. Look at `examples/tiny_experiments/` for patterns
4. Use the Task tool for complex searches instead of manual grep

## Development Workflow That Works

### 1. Initial Setup
1. Create workspace structure with proper `base_directories.yaml`
2. Create minimal `meta.yaml` and `project.yaml`
3. Test basic project loading before adding complexity

### 2. Template Development
1. Start with simple template that extends base types
2. Use split document approach for complex inheritance
3. Test each component incrementally
4. Add truncation settings early to avoid OOM

### 3. Architecture Design
1. Choose either relative (ALiBi) OR absolute (sinusoidal) positional encoding
2. Verify component compatibility before combining
3. Start with standard components, then customize
4. Always check parameter count to verify architecture

### 4. Training & Debugging
1. Start with max_steps: 5 for quick validation
2. Add truncation to prevent OOM
3. Use single GPU initially, scale to multi-GPU later
4. Monitor logs and Tensorboard for issues

### 5. Experimentation
1. Create separate configs for each experiment
2. Use same base template, override specific parameters
3. Document hypothesis before running experiments
4. Compare results systematically

## Anti-Patterns to Avoid

### Template Design
- ❌ Complex Jinja2 inheritance with circular dependencies
- ❌ Mixing relative and absolute positional encoding
- ❌ Creating new files when editing existing ones would work
- ❌ Adding forgather to sys.path in notebooks (assume it's installed)

### Training Setup
- ❌ Starting with full dataset without truncation
- ❌ Using explicit Project constructor parameters unnecessarily  
- ❌ Setting RANK=0 as permanent solution instead of fixing root cause
- ❌ Running long training without testing short runs first

### Code Organization
- ❌ Creating documentation files proactively (only when requested)
- ❌ Using complex inheritance when simple configuration override works
- ❌ Hardcoding paths instead of using template variables

## Success Patterns

### Template Design
- ✅ Use split document approach for complex templates
- ✅ Start simple, add complexity incrementally
- ✅ Test template loading before adding custom components
- ✅ Use existing components as much as possible

### Experimentation  
- ✅ Clear hypothesis before running experiments
- ✅ Controlled comparisons (change one thing at a time)
- ✅ Document results immediately after training
- ✅ Use systematic naming for experiment configs

### Development Process
- ✅ Test basic functionality before adding features
- ✅ Use truncation for initial experiments
- ✅ Validate architecture before training
- ✅ Monitor training progress and stop early if issues arise

## Final Notes

The key to successful external workspace development is:
1. **Start simple** - get basic setup working first
2. **Test incrementally** - validate each step before proceeding  
3. **Use proven patterns** - follow existing examples closely
4. **Document as you go** - capture lessons while they're fresh
5. **Think systematically** - plan experiments with clear hypotheses

Remember: Forgather's power comes from systematic template-based configuration management. Embrace the template system rather than fighting it, and use the extensive existing templatelib as much as possible.