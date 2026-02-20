# Contributing to SentinelMesh

First off, **thank you** for considering contributing to SentinelMesh! 🎉

It's people like you that make SentinelMesh the best AI Operating System in the world.

---

## 🌟 Ways to Contribute

### 1. Report Bugs 🐛
Found a bug? Please create an issue with:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs

### 2. Suggest Features 💡
Have an idea? We'd love to hear it!
- Check existing issues first
- Describe the problem it solves
- Explain your proposed solution
- Consider implementation details

### 3. Improve Documentation 📚
Documentation is crucial!
- Fix typos and clarity
- Add examples
- Write tutorials
- Create diagrams
- Translate to other languages

### 4. Write Code 💻
Ready to dive in?
- Fix bugs
- Implement features
- Optimize performance
- Add integrations
- Create plugins

### 5. Help Others 🤝
Community is everything!
- Answer questions on Discord
- Review pull requests
- Share your success stories
- Create tutorials and blog posts

---

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/sentinelmesh.git
cd sentinelmesh
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
# Feature branch
git checkout -b feature/amazing-feature

# Bug fix branch
git checkout -b fix/bug-description
```

### 4. Make Your Changes

```bash
# Write code
# Add tests
# Update documentation

# Run tests
pytest tests/

# Run linters
black .
flake8 .
mypy .

# Check formatting
pre-commit run --all-files
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "feat: add amazing feature"

# Use conventional commits:
# feat: new feature
# fix: bug fix
# docs: documentation
# style: formatting
# refactor: code restructuring
# test: adding tests
# chore: maintenance
```

### 6. Push and Create PR

```bash
git push origin feature/amazing-feature
```

Then create a Pull Request on GitHub with:
- Clear title
- Description of changes
- Link to related issues
- Screenshots (if UI changes)
- Test coverage

---

## 📋 Pull Request Guidelines

### Before Submitting

- [ ] Code follows project style
- [ ] Tests pass (`pytest tests/`)
- [ ] New code has tests
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with main

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Related Issues
Closes #123

## Testing
How was this tested?

## Screenshots (if applicable)

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guide
```

---

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest

# Specific test
pytest tests/test_memory.py

# With coverage
pytest --cov=core tests/

# Verbose
pytest -v
```

### Writing Tests

```python
import pytest
from core.memory import MemoryManager

@pytest.mark.asyncio
async def test_memory_store():
    """Test memory storage."""
    memory = MemoryManager()
    
    await memory.store_interaction(
        user_id="test_user",
        session_id="test_session",
        prompt="Hello",
        response="Hi there!"
    )
    
    result = await memory.recall_context("test_user", "Hello")
    assert len(result["memories"]) > 0
```

---

## 📝 Code Style

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Good
async def process_request(
    prompt: str,
    user_id: str,
    options: Optional[Dict] = None
) -> Response:
    """Process user request with context."""
    if options is None:
        options = {}
    
    # Process
    result = await router.route(prompt)
    return result


# Bad
async def process_request(prompt,user_id,options=None):
    if options==None: options={}
    result=await router.route(prompt)
    return result
```

### Key Principles

1. **Clear over clever**
2. **Explicit over implicit**
3. **Type hints always**
4. **Docstrings for public APIs**
5. **Comments for complex logic**

### Tools We Use

- **Black:** Auto-formatting
- **Flake8:** Linting
- **MyPy:** Type checking
- **Pytest:** Testing
- **Pre-commit:** Git hooks

---

## 🏗️ Project Structure

```
sentinelmesh/
├── core/                   # Core functionality
│   ├── memory/            # Memory system
│   ├── streaming/         # Streaming
│   ├── cache/             # Semantic cache
│   ├── functions/         # Function calling
│   ├── workflows/         # Workflow engine
│   ├── integrations/      # Integration manager
│   ├── guardrails/        # Safety system
│   ├── plugins/           # Plugin system
│   ├── collaboration/     # Team features
│   └── voice/             # Voice interface
├── tests/                 # Test suite
├── docs/                  # Documentation
├── app.py                 # FastAPI application
├── config.py              # Configuration
└── requirements.txt       # Dependencies
```

---

## 🐛 Bug Report Template

```markdown
**Describe the bug**
Clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Do this...
2. Then do that...
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10]
- SentinelMesh: [e.g., 3.0.0]
- Deployment: [e.g., Docker, Railway]

**Logs**
```
Paste relevant logs here
```

**Additional context**
Any other relevant information.
```

---

## 💡 Feature Request Template

```markdown
**Is your feature request related to a problem?**
Clear description of the problem.

**Describe the solution you'd like**
Clear description of what you want to happen.

**Describe alternatives you've considered**
Other solutions you've thought about.

**Use case**
How would this feature be used?

**Additional context**
Mockups, examples, etc.
```

---

## 🎯 Development Priorities

### High Priority
- Bug fixes
- Security issues
- Performance improvements
- Documentation gaps

### Medium Priority
- New integrations
- Feature enhancements
- Developer experience
- Test coverage

### Low Priority
- Nice-to-have features
- Cosmetic improvements
- Refactoring (without clear benefit)

---

## 🔒 Security Issues

**DO NOT** create public issues for security vulnerabilities!

Instead:
1. Email: security@sentinelmesh.ai
2. Include:
   - Vulnerability description
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)
3. We'll respond within 24 hours
4. Fix will be prioritized
5. Credit given in security advisory

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Recognition

Contributors are recognized:
- In README.md
- In release notes
- On our website (coming soon)
- With special Discord role

Top contributors get:
- Direct access to maintainers
- Early access to features
- Swag (when available)
- References/recommendations

---

## 💬 Communication

### Discord
Join our Discord: [Link]
- #general - General discussion
- #help - Get help
- #development - Development chat
- #contributions - Discuss contributions

### GitHub Discussions
For longer conversations:
- Ideas and feature requests
- Q&A
- Show and tell

### Office Hours
Every Friday, 3 PM UTC
- Ask questions
- Discuss roadmap
- Get help with PRs

---

## 📚 Resources

- **Documentation:** All *_GUIDE.md files
- **API Reference:** /docs endpoint
- **Examples:** examples/ directory
- **Discord:** [Link]
- **Twitter:** @SentinelMesh

---

## ✅ Checklist for First-Time Contributors

- [ ] Read this guide
- [ ] Join Discord
- [ ] Fork repository
- [ ] Set up development environment
- [ ] Find "good first issue"
- [ ] Ask questions if stuck
- [ ] Submit PR
- [ ] Celebrate! 🎉

---

## 🌍 Code of Conduct

Be kind, be respectful, be collaborative.

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.

---

**Thank you for making SentinelMesh better! 🚀**

Every contribution, no matter how small, makes a difference.

Questions? Ask on Discord or create a discussion on GitHub!
