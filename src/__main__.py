# ABOUTME: Command-line entry point for the "call me maybe" function-calling
# ABOUTME: tool. Wires together file I/O, the LLM SDK, and the constrained-
# ABOUTME: decoding pipeline.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.function_caller import resolve_all
from src.io_utils import (
    ProjectIOError,
    load_function_definitions,
    load_prompts,
    save_results,
)
from src.vocab import load_vocab

DEFAULT_FUNCTIONS_DEFINITION = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")
DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description=(
            "Translate natural-language prompts into structured function "
            "calls using constrained decoding over a small local LLM."
        ),
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS_DEFINITION,
        help=(
            "Path to the function definitions JSON file "
            f"(default: {DEFAULT_FUNCTIONS_DEFINITION})."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Path to the natural-language prompts JSON file "
            f"(default: {DEFAULT_INPUT})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Path to write the resulting JSON file to "
            f"(default: {DEFAULT_OUTPUT})."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=(
            "Hugging Face model identifier to use "
            f"(default: {DEFAULT_MODEL_NAME})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        function_definitions = load_function_definitions(
            args.functions_definition
        )
        prompts = load_prompts(args.input)
    except ProjectIOError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        # Imported lazily: this pulls in torch/transformers, which are
        # heavy and only actually needed once the input files are known
        # to be valid.
        from llm_sdk import Small_LLM_Model
    except ImportError as exc:
        print(
            f"Error: could not import llm_sdk ({exc}). Is it installed?",
            file=sys.stderr,
        )
        return 1

    try:
        llm = Small_LLM_Model(model_name=args.model)
        vocab_path = llm.get_path_to_vocab_file()
        id_to_text = load_vocab(vocab_path)
    except Exception as exc:  # noqa: BLE001 - model errors must not crash CLI
        print(
            f"Error: could not load the language model {args.model!r}: {exc}",
            file=sys.stderr,
        )
        return 1

    results = resolve_all(llm, prompts, function_definitions, id_to_text)

    try:
        save_results(args.output, results)
    except ProjectIOError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(results)} result(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
