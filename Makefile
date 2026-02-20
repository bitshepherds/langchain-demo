.PHONY: install run run-basic run-tracing pytest test-query run-evaluation dspy clean help

help:
	@echo "LangGraph Hello World - Available commands:"
	@echo ""
	@echo "=== Setup ==="
	@echo "  make install      Install dependencies (creates .venv)"
	@echo "  make ollama      Start Ollama server"
	@echo ""
	@echo "=== Approaches ==="
	@echo "  make run-basic       Approach 1: Basic agent (LangChain + LangGraph)"
	@echo "  make run-tracing    Approach 2: Agent with LangSmith tracing"
	@echo "  make pytest         Approach 3: Unit tests (Pytest)"
	@echo "  make run-evaluation Approach 4: LLM-as-judge evaluation"
	@echo "  make dspy           Approach 5: DSPy prompt optimization"
	@echo ""
	@echo "=== Other ==="
	@echo "  make test-query   Run a quick test query"
	@echo "  make upload       Upload golden dataset to LangSmith"
	@echo "  make clean        Remove .venv"
	@echo ""

install:
	uv venv && uv pip install -r requirements.txt

# Approach 1: Basic Agent
run-basic:
	@echo "Running basic LangGraph agent..."
	@source .venv/bin/activate && python main.py

# Approach 2: Tracing (same as basic, but with LangSmith enabled)
run-tracing:
	@echo "Running agent with LangSmith tracing enabled..."
	@source .venv/bin/activate && python main.py

# Approach 3: Testing
pytest:
	@source .venv/bin/activate && pytest tests/ -v

test-query:
	@source .venv/bin/activate && echo "What is 25 + 17?" | python main.py

# Approach 4: Evaluation
upload:
	@source .venv/bin/activate && python scripts/upload_dataset.py

run-evaluation:
	@source .venv/bin/activate && python scripts/run_evaluation.py

# Approach 5: DSPy
dspy:
	@source .venv/bin/activate && python scripts/dspy_example.py

# Legacy alias
run: run-basic

ollama:
	ollama serve

clean:
	rm -rf .venv
