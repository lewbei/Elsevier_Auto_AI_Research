# Contributing to Elsevier Auto AI Research

Thank you for your interest in contributing to this automated AI research pipeline! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/Elsevier_Auto_AI_Research.git
   cd Elsevier_Auto_AI_Research
   ```
3. **Set up your development environment** following the [installation instructions](README.md#installation)
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🧪 Development Setup

### Environment Setup
1. Install dependencies:
   ```bash
   cmd.exe /C "python -m pip install -r requirements.txt --disable-pip-version-check"
   ```

2. Set up your `.env` file with required API keys (see README.md)

3. Verify your setup:
   ```bash
   # Compile all sources
   cmd.exe /C "python -c \"import sys,compileall; sys.exit(0 if compileall.compile_dir('.', force=True, quiet=1) else 1)\""
   
   # Run tests
   cmd.exe /C "python -m pytest -q"
   ```

## 📝 Code Style and Standards

### Python Code Style
- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Use meaningful variable and function names

### Documentation
- Update documentation for any new features
- Include docstrings with examples for public APIs
- Update README.md if adding new functionality
- Add comments for complex logic

### Testing
- Write tests for new functionality
- Ensure tests pass before submitting PR
- Tests should not require external API access
- Keep tests fast and focused

## 🔧 Areas for Contribution

### High Priority
- **Enhanced Error Handling**: Improve robustness of API interactions
- **Additional Datasets**: Support for more ML datasets
- **Visualization Improvements**: Better charts and dashboards
- **Performance Optimization**: Reduce API calls and improve caching

### Medium Priority
- **Documentation**: Improve code documentation and examples
- **Testing**: Expand test coverage
- **Configuration**: More granular configuration options
- **Logging**: Enhanced logging and debugging features

### Advanced Features
- **Multi-language Support**: Support for non-English papers
- **Custom Models**: Support for different LLM providers
- **Distributed Execution**: Support for parallel experiment execution
- **Integration**: Integration with other research tools

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Environment**: OS, Python version, and dependencies
3. **Steps to Reproduce**: Detailed steps to recreate the issue
4. **Expected Behavior**: What you expected to happen
5. **Actual Behavior**: What actually happened
6. **Configuration**: Relevant environment variables (sanitize API keys!)
7. **Logs**: Any relevant log output or error messages

## 🚀 Feature Requests

For feature requests, please include:

1. **Use Case**: Why this feature would be useful
2. **Description**: Detailed description of the proposed feature
3. **Implementation Ideas**: Any thoughts on how it could be implemented
4. **Alternatives**: Any alternative solutions you've considered

## 📋 Pull Request Process

1. **Create an Issue**: For significant changes, create an issue first to discuss
2. **Follow Code Style**: Ensure your code follows the project's style guidelines
3. **Write Tests**: Add or update tests for your changes
4. **Update Documentation**: Update relevant documentation
5. **Test Thoroughly**: Run the full test suite and verify functionality
6. **Write Clear Commits**: Use descriptive commit messages
7. **Submit PR**: Submit your pull request with a clear description

### Pull Request Template

```markdown
## Description
Brief description of what this PR does.

## Changes Made
- List of specific changes
- Each change on a separate line

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Documentation
- [ ] Code comments updated
- [ ] README.md updated (if needed)
- [ ] Documentation updated (if needed)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Changes are backwards compatible
- [ ] No breaking changes (or clearly documented)
```

## 🏗️ Development Guidelines

### Code Organization
- Keep modules focused and cohesive
- Use appropriate separation of concerns
- Follow existing file naming conventions
- Place utilities in appropriate modules

### API Integration
- Always handle API failures gracefully
- Implement appropriate retry logic
- Respect rate limits
- Cache responses when appropriate
- Never commit API keys to version control

### Windows Compatibility
- Use `cmd.exe /C "python ..."` for subprocess calls
- Handle Windows path separators correctly
- Test on Windows when possible
- Avoid Unix-specific commands

### Performance Considerations
- Minimize API calls through caching
- Use appropriate data structures
- Avoid unnecessary file I/O
- Consider memory usage for large datasets

## 🔒 Security Guidelines

- **Never commit secrets**: API keys, tokens, or credentials
- **Sanitize logs**: Remove sensitive information from log outputs
- **Validate inputs**: Sanitize all external inputs and API responses
- **Use environment variables**: For all configuration and secrets

## 📞 Getting Help

- **Documentation**: Check [AGENTS.md](AGENTS.md) for detailed setup instructions
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Code Review**: Ask for code review on complex changes

## 🙏 Recognition

Contributors will be recognized in:
- GitHub contributors list
- Release notes for significant contributions
- Documentation acknowledgments

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same [MIT License](LICENSE) that covers the project.

---

Thank you for contributing to make AI research more accessible and automated! 🤖✨