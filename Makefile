export UV_CACHE_DIR := $(PWD)/.uv_cache

NAME = call_me_maybe

all: install

install:
	uv sync

# Regra extra para instalares o SDK agora mesmo
add-sdk:
	uv add ./llm_sdk

run:
	uv run python -m src

clean:
	rm -rf .venv
	rm -rf .uv_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

fclean: clean

re: fclean all

lint:
	uv run mypy src
	uv run flake8 src

.PHONY: all install add-sdk run clean fclean re
