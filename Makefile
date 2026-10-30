GOINFRE_DIR = /home/$(USER)/goinfre/call_me_maybe

UV_CACHE    = $(GOINFRE_DIR)/uv-cache
HF_CACHE    = $(GOINFRE_DIR)/huggingface

EXPORT_UV_CACHE = export UV_CACHE_DIR="$(GOINFRE_DIR)/uv-cache"
EXPORT_HF_HOME  = export HF_HOME="$(GOINFRE_DIR)/huggingface"
EXPORT_VENV     = export UV_PROJECT_ENVIRONMENT="$(GOINFRE_DIR)/.venv"

install:
	uv sync

setup_goinfre:
	mkdir -p $(UV_CACHE)
	mkdir -p $(HF_CACHE)

	@echo "echo '$(EXPORT_UV_CACHE)' >> ~/.zshrc"
	@grep -qxF '$(EXPORT_UV_CACHE)' ~/.zshrc || echo '$(EXPORT_UV_CACHE)' >> ~/.zshrc

	@echo "echo '$(EXPORT_HF_HOME)' >> ~/.zshrc"
	@grep -qxF '$(EXPORT_HF_HOME)' ~/.zshrc || echo '$(EXPORT_HF_HOME)' >> ~/.zshrc

	@echo "echo '$(EXPORT_VENV)' >> ~/.zshrc"
	@grep -qxF '$(EXPORT_VENV)' ~/.zshrc || echo '$(EXPORT_VENV)' >> ~/.zshrc

	@echo ""
	@echo "If your current shell is listed below, follow the corresponding steps:"
	@echo "┌─────────┬──────────────────────────────────────────────┐"
	@echo "│ Shell   │ Commands                                     │"
	@echo "├─────────┼──────────────────────────────────────────────┤"
	@echo "│ zsh     │ ✔ Nothing to do                              │"
	@echo "├─────────┼──────────────────────────────────────────────┤"
	@echo "│ fish    │ zsh                                          │"
	@echo "│         │ source ~/.zshrc                              │"
	@echo "│         │ fish                                         │"
	@echo "├─────────┼──────────────────────────────────────────────┤"
	@echo "│ bash    │ zsh                                          │"
	@echo "│         │ source ~/.zshrc                              │"
	@echo "│         │ bash                                         │"
	@echo "└─────────┴──────────────────────────────────────────────┘"
	@echo ""

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	@echo "Removing temporary files and caches..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf \
	.mypy_cache \
	src/.mypy_cache \
	llm_sdk/.mypy_cache \

fclean_goinfre: clean
	@echo "Removing the 'goinfre' project directory..."
	rm -rf $(GOINFRE_DIR)

fclean: clean
	@echo "Removing files created without setup_goinfre..."
	@rm -rf \
	.venv \
	~/.cache/uv \
	~/.cache/huggingface \
	~/.cache/transformers \
	~/.cache/torch

re_goinfre: fclean_goinfre install

re: fclean install

lint:
	@flake8 src
	@mypy src \
	--warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--follow-imports=skip \
	--disallow-untyped-defs \
	--check-untyped-defs

.PHONY: \
	re \
	run \
	lint \
	debug \
	clean \
	fclean \
	install \
	re_goinfre \
	setup_goinfre \
	fclean_goinfre \