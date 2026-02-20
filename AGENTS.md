# Agent Development Guidelines

## This Project

- **Purpose**: Learn LangChain/LangGraph/LangSmith/DSPy ecosystem
- **Language**: Python 3.10+
- **Model**: Ollama (llama3.2) - runs locally, free

## Setup

```bash
# Install dependencies
make install

# Start Ollama (separate terminal)
make ollama
```

## Approaches (5 Ways to Build LLM Apps)

| # | Approach | Command | Files |
|---|----------|---------|-------|
| 1 | Basic Agent | `make run` | `main.py`, `tools.py` |
| 2 | Tracing | Enable LangSmith | `.env` |
| 3 | Testing | `make pytest` | `tests/test_agent.py` |
| 4 | LLM-as-Judge | `python scripts/run_evaluation.py` | `scripts/run_evaluation.py` |
| 5 | DSPy | `make dspy` | `scripts/dspy_example.py` |

## Commands

```bash
# Run agent
make run

# Run tests
make pytest

# Run LLM-as-judge evaluation
python scripts/upload_dataset.py
python scripts/run_evaluation.py

# Run DSPy optimization
make dspy

# View traces
https://smith.langchain.com
```

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Basic LangGraph agent |
| `tools.py` | Tool definitions |
| `tests/test_agent.py` | Pytest cases |
| `scripts/run_evaluation.py` | LLM-as-judge |
| `scripts/dspy_example.py` | DSPy optimization |
| `pyproject.toml` | Test config |
