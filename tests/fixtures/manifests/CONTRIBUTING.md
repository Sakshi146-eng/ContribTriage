# Contributing Guide

Thank you for your interest in contributing to this project!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/repo.git`
3. Create a feature branch: `git checkout -b feature/my-feature`

## Development Setup

Install dependencies:

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v --cov=src
```

## Code Style

We use `ruff` for linting and formatting. Run before committing:

```bash
ruff check . --fix
ruff format .
```

## Pull Request Guidelines

- All PRs must include tests
- Keep commits atomic and well-described
- Update documentation if behaviour changes
- Reference the related issue number in your PR description
