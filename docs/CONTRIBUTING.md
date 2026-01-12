# Contributing to DocIntel

Thank you for considering contributing to DocIntel! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Style Guide](#style-guide)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all.

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- OpenAI API key
- Basic understanding of RAG systems (helpful but not required)

### Areas for Contribution

We welcome contributions in:

**Code**
- Bug fixes
- New features
- Performance improvements
- Test coverage

**Documentation**
- Improving existing docs
- Adding tutorials
- Translating documentation
- Creating examples

**Design**
- UI/UX improvements
- Architecture suggestions
- Performance optimizations

---

## Development Setup

### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/docintel.git
cd docintel
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"
```

### 4. Set Up Environment
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OPENAI_API_KEY
```

### 5. Verify Setup
```bash
# Run tests
pytest tests/ -v

# Check imports
python -c "from src.retrieval.rag_pipeline import RAGPipeline; print('✅ Setup complete!')"
```

---

## How to Contribute

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

**Branch naming conventions:**
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/improvements

### 2. Make Changes

- Write clean, readable code
- Follow the style guide
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_your_changes.py -v

# Check code style
black src/ tests/ --check
flake8 src/ tests/

# Check test coverage
pytest tests/ --cov=src --cov-report=html
```

### 4. Commit Changes
```bash
git add .
git commit -m "feat: add new feature"
```

**Commit message format:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

---

## Style Guide

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

**Formatting**
- Use `black` for code formatting
- Line length: 88 characters (black default)
- Use 4 spaces for indentation

**Naming Conventions**
```python
# Classes: PascalCase
class DocumentParser:
    pass

# Functions and variables: snake_case
def parse_document(file_path):
    chunk_size = 512

# Constants: UPPER_SNAKE_CASE
MAX_CHUNK_SIZE = 1000

# Private methods/variables: _leading_underscore
def _internal_helper():
    pass
```

**Type Hints**
Always use type hints:
```python
def process_text(text: str, max_length: int = 100) -> List[str]:
    """Process text into chunks."""
    ...
```

**Docstrings**
Use Google-style docstrings:
```python
def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Similarity score between 0 and 1
        
    Raises:
        ValueError: If vectors have different dimensions
        
    Example:
        >>> sim = calculate_similarity([1, 0], [1, 0])
        >>> print(sim)
        1.0
    """
    ...
```

**Imports**
Organize imports:
```python
# Standard library
import os
from pathlib import Path
from typing import List, Dict

# Third-party
import numpy as np
from fastapi import FastAPI

# Local
from src.utils.config import settings
from src.ingestion.parser import DocumentParser
```

### Code Quality Tools

**Black** (formatting)
```bash
black src/ tests/ streamlit_app/
```

**Flake8** (linting)
```bash
flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203
```

**isort** (import sorting)
```bash
isort src/ tests/ --profile black
```

---

## Testing

### Writing Tests

**Test Structure**
```python
import pytest
from src.module import YourClass

class TestYourClass:
    """Test suite for YourClass."""
    
    @pytest.fixture
    def instance(self):
        """Create test instance."""
        return YourClass()
    
    def test_basic_functionality(self, instance):
        """Test basic functionality."""
        result = instance.method()
        assert result is not None
    
    def test_edge_case(self, instance):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            instance.method(invalid_input)
```

**Test Categories**

Use markers:
```python
@pytest.mark.unit
def test_unit():
    """Unit test."""
    pass

@pytest.mark.integration
def test_integration():
    """Integration test."""
    pass

@pytest.mark.slow
def test_slow_operation():
    """Slow test."""
    pass
```

**Running Tests**
```bash
# All tests
pytest tests/ -v

# Specific category
pytest tests/ -v -m unit
pytest tests/ -v -m "not slow"

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

Aim for:
- **80%+ overall coverage**
- **100% coverage for critical paths** (RAG pipeline, API endpoints)
- **All public methods tested**

---

## Pull Request Process

### Before Submitting

1. **Update Documentation**
   - Update README if adding features
   - Add docstrings to new functions
   - Update API.md for API changes

2. **Add Tests**
   - Unit tests for new functionality
   - Integration tests for workflows
   - Update existing tests if behavior changed

3. **Run Full Test Suite**
```bash
   pytest tests/ -v
   black src/ tests/ --check
   flake8 src/ tests/
```

4. **Update CHANGELOG**
   Add entry to `CHANGELOG.md`:
```markdown
   ### [Unreleased]
   #### Added
   - New feature description
   
   #### Fixed
   - Bug fix description
```

### PR Template

When creating a PR, include:
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests passing
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] Tests added
- [ ] No breaking changes (or documented)

## Screenshots (if applicable)
Add screenshots for UI changes

## Related Issues
Closes #123
```

### Review Process

1. **Automated Checks**
   - Tests must pass
   - Code style checks must pass
   - No merge conflicts

2. **Code Review**
   - At least one maintainer approval required
   - Address all review comments
   - Be respectful and open to feedback

3. **Merge**
   - Squash commits for clean history
   - Maintainer will merge after approval

---

## Reporting Bugs

### Before Reporting

1. Check if issue already exists
2. Try latest version of code
3. Verify it's not a configuration issue

### Bug Report Template
```markdown
**Describe the bug**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen

**Actual behavior**
What actually happened

**Screenshots**
If applicable

**Environment:**
- OS: [e.g. Windows 11]
- Python version: [e.g. 3.11.5]
- DocIntel version: [e.g. 1.0.0]

**Additional context**
Any other relevant information
```

---

## Suggesting Enhancements

### Enhancement Template
```markdown
**Is your feature request related to a problem?**
Clear description of the problem

**Describe the solution you'd like**
Clear description of desired behavior

**Describe alternatives you've considered**
Other solutions you've thought about

**Additional context**
Mockups, examples, etc.
```

---

## Development Guidelines

### Performance

- Profile code before optimizing
- Use batching for API calls
- Cache expensive operations
- Consider memory usage for large documents

### Security

- Never commit API keys or secrets
- Validate all user input
- Sanitize file uploads
- Follow OWASP guidelines

### Error Handling
```python
# Good: Specific exception handling
try:
    result = process_document(file_path)
except FileNotFoundError:
    logger.error(f"File not found: {file_path}")
    raise
except ValueError as e:
    logger.error(f"Invalid file format: {e}")
    raise

# Bad: Bare except
try:
    result = process_document(file_path)
except:
    pass  # Silently fails!
```

### Logging
```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Detailed info for debugging")
logger.info("General information")
logger.warning("Warning about potential issue")
logger.error("Error occurred")
logger.critical("Critical error")
```

---

## Getting Help

**Questions?**
- GitHub Discussions: [github.com/yourusername/docintel/discussions](https://github.com/yourusername/docintel/discussions)
- Email: your.email@example.com

**Found a security issue?**
Please email security@yourdomain.com (do not create public issue)

---

## Recognition

Contributors will be:
- Listed in README.md
- Mentioned in release notes
- Given credit in documentation

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to DocIntel! 🎉