.PHONY: test test-cpu test-gpu

test-cpu:
	@python -m pytest --version >/dev/null 2>&1 || (echo "pytest is not installed for this Python. Install with: pip install pytest"; exit 1)
	python -m pytest -m "not gpu"

test-gpu:
	@python -m pytest --version >/dev/null 2>&1 || (echo "pytest is not installed for this Python. Install with: pip install pytest"; exit 1)
	python -m pytest

test: test-cpu
