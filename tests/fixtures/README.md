# Sample Project

A fixture repository for ContribTriage ingestion tests.

## Installation

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Or with uv for faster installs:

```bash
uv pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting pull requests.

All contributions must:
- Include unit tests
- Pass the existing test suite
- Follow the code style guide

## Architecture

This project uses a pipeline architecture:

1. **Ingestion** — parses source files into a knowledge graph
2. **Processing** — applies transformations via the `DataProcessor` class
3. **Output** — writes results to the configured output directory

## Known Issues

- Large files (>100MB) may cause memory pressure during ingestion
- The `DataProcessor.transform()` method is not thread-safe
