# Contributing to Forgather

Thank you for your interest in contributing to Forgather! This guide will help you get started with development and contributions.

## Quick Links

- **[Development Setup](development-setup.md)** - Setting up your development environment
- **[Adding Trainers](adding-trainers.md)** - How to add new trainer types
- **[Adding Models](adding-models.md)** - How to add new model architectures
- **[Documentation](documentation.md)** - Writing and maintaining documentation

## Getting Started

### 1. Development Environment
```bash
# Clone the repository
git clone https://github.com/anthropics/forgather.git
cd forgather

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

### 2. Understanding the Codebase
- **Core framework**: `src/forgather/`
- **ML components**: `src/forgather/ml/`
- **Template library**: `templatelib/`
- **Examples**: `examples/`
- **Tests**: `tests/`

### 3. Development Workflow
1. **Fork and clone** the repository
2. **Create a feature branch** for your changes
3. **Write tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with clear description

## Types of Contributions

### Bug Fixes
- Fix issues reported in GitHub
- Add regression tests
- Update documentation if needed

### New Features
- Trainer implementations
- Model architectures
- Configuration enhancements
- Tool improvements

### Documentation
- API documentation
- Tutorials and guides
- Example projects
- Typo fixes and clarifications

### Testing
- Unit tests
- Integration tests
- Performance benchmarks
- Example validation

## Development Guidelines

### Code Style
- **Python**: Follow PEP 8, use Black for formatting
- **YAML**: Consistent indentation, clear naming
- **Documentation**: Clear, concise, with examples

### Testing Requirements
- **Unit tests**: For all new functionality
- **Integration tests**: For trainer and model additions
- **Documentation tests**: Ensure examples work

### Performance Considerations
- **Memory efficiency**: Especially for large models
- **Training speed**: Minimize overhead
- **Scalability**: Consider distributed training impact

## Contribution Workflow

### 1. Planning
- **Check existing issues** for similar work
- **Create an issue** for discussion if needed
- **Get feedback** before starting large changes

### 2. Development
```bash
# Create feature branch
git checkout -b feature/my-new-feature

# Make changes
# ... develop and test ...

# Commit changes
git add .
git commit -m "Add new feature: description"

# Push to your fork
git push origin feature/my-new-feature
```

### 3. Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Test documentation examples
pytest tests/docs/
```

### 4. Pull Request
- **Clear title** describing the change
- **Detailed description** with motivation and impact
- **Link to related issues** if applicable
- **Test results** and validation

## Specific Contribution Areas

### Adding New Trainers
See [Adding Trainers](adding-trainers.md) for:
- Trainer interface requirements
- Configuration integration
- Testing requirements
- Documentation standards

### Adding New Models
See [Adding Models](adding-models.md) for:
- Model architecture patterns
- Template creation
- Component integration
- Validation requirements

### Improving Documentation
See [Documentation Guidelines](documentation.md) for:
- Style and formatting
- Example requirements
- Reference documentation
- User guide standards

## Community Guidelines

### Code of Conduct
- **Be respectful** and inclusive
- **Provide constructive feedback**
- **Help others learn** and contribute
- **Follow project standards**

### Communication
- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, general discussion
- **Pull Request Reviews**: Constructive technical feedback

### Recognition
Contributors are recognized through:
- **Contributor list** in documentation
- **Release notes** for significant contributions
- **Maintainer status** for ongoing contributors

## Getting Help

### Development Questions
- **GitHub Discussions** for general questions
- **Issue comments** for specific problems
- **Discord/Slack** for real-time discussion (if available)

### Code Review Process
- **Automated checks** must pass
- **Maintainer review** required for changes
- **Documentation review** for user-facing changes
- **Performance testing** for training components

### Release Process
- **Feature freeze** before releases
- **Testing period** with community feedback
- **Documentation updates** for new features
- **Migration guides** for breaking changes

## Development Resources

### Tools and Scripts
- **`scripts/dev_setup.py`** - Development environment setup
- **`scripts/run_tests.py`** - Comprehensive testing
- **`scripts/format_code.py`** - Code formatting
- **`scripts/build_docs.py`** - Documentation building

### Reference Materials
- **[Architecture Overview](../core-concepts/)** - Framework design
- **[API Reference](../reference/)** - Complete API documentation
- **[Examples](../examples/)** - Working code examples

### Testing Infrastructure
- **GitHub Actions** for CI/CD
- **Docker containers** for reproducible testing
- **Performance benchmarks** for regression testing

---

*Thank you for contributing to Forgather! Your efforts help make ML research more systematic, reproducible, and accessible.*