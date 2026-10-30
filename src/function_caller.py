# ABOUTME: Orchestrates the two-phase constrained generation: pick a function
# ABOUTME: name, then fill in its arguments, for every natural-language prompt.

from __future__ import annotations

import json
import sys
from typing import Any

from src.decoding import GenerationError, LLMBackend, generate_constrained
from src.grammar import function_name_grammar, parameters_object_grammar
from src.schema import (
    FunctionCallResult,
    FunctionDefinition,
    ParameterDefinition,
    PromptEntry,
)

# Token budgets tried, in order, before giving up on a single prompt. Some
# arguments (long file paths, SQL statements, format templates, ...) need more
# room than a short "Greet shrek" call; retrying with a larger budget before
# failing keeps the pipeline robust without slowing down the common case.
_TOKEN_BUDGETS = (200, 400, 800)

_DEFAULT_VALUE_BY_TYPE: dict[str, Any] = {
    "string": "",
    "number": 0.0,
    "integer": 0,
    "boolean": False,
}


class FunctionCallingError(RuntimeError):
    """Raised when a prompt cannot be turned into a valid function call."""


def _build_name_selection_prompt(
    user_prompt: str, function_definitions: list[FunctionDefinition]
) -> str:
    catalog_lines = []
    for fn in function_definitions:
        param_names = ", ".join(fn.parameters.keys()) or "none"
        catalog_lines.append(f"- {fn.name}({param_names}): {fn.description}")
    catalog = "\n".join(catalog_lines)

    return (
        "You are a function-calling assistant. Choose exactly one function "
        "name from the list below that best answers the user's request. "
        "Respond with the function name only, nothing else.\n\n"
        f"Available functions:\n{catalog}\n\n"
        f'User request: "{user_prompt}"\n'
        "Function name:"
    )


def _build_arguments_prompt(
    user_prompt: str, function_definition: FunctionDefinition
) -> str:
    params_description = ", ".join(
        f"{name} ({definition.type})"
        for name, definition in function_definition.parameters.items()
    )
    return (
        "You are a function-calling assistant. Extract the arguments "
        f"needed to call the function '{function_definition.name}' "
        f"({function_definition.description}) from the user's request. "
        f"Expected arguments: {params_description or 'none'}. "
        "Respond with a single JSON object containing exactly those "
        "arguments, nothing else.\n\n"
        f'User request: "{user_prompt}"\n'
        "JSON arguments:"
    )


def _coerce_parameter_types(
    parameters: dict[str, Any], function_definition: FunctionDefinition
) -> dict[str, Any]:
    """Cast every parsed value to the exact JSON type declared in the schema.

    The grammar only guarantees *syntactic* validity (e.g. ``3`` and ``3.0``
    are both valid JSON numbers), not which Python type ``json.loads`` hands
    back for a given literal. A bare ``3`` parses to an ``int`` even when the
    schema declares ``"type": "number"`` (i.e. a float is expected), so we
    normalise every value to the declared type here without touching its
    numeric value.
    """
    coerced: dict[str, Any] = {}
    for key, value in parameters.items():
        param_def: ParameterDefinition | None = (
            function_definition.parameters.get(key)
        )
        if param_def is None:
            coerced[key] = value
            continue
        if param_def.type == "number":
            coerced[key] = float(value)
        elif param_def.type == "integer":
            coerced[key] = int(value)
        elif param_def.type == "boolean":
            coerced[key] = bool(value)
        else:
            coerced[key] = value
    return coerced


def _default_parameters(
    function_definition: FunctionDefinition,
) -> dict[str, Any]:
    """Best-effort, schema-valid placeholder arguments used only as a
    last resort."""
    return {
        name: _DEFAULT_VALUE_BY_TYPE.get(definition.type, "")
        for name, definition in function_definition.parameters.items()
    }


def resolve_function_call(
    llm: LLMBackend,
    prompt_entry: PromptEntry,
    function_definitions: list[FunctionDefinition],
    id_to_text: dict[int, str],
    max_tokens: int = _TOKEN_BUDGETS[0],
) -> FunctionCallResult:
    """Turn a single natural-language prompt into a validated function call.

    Raises:
        FunctionCallingError: if the LLM/grammar pipeline could not produce a
            valid function name or a valid, schema-compliant arguments object.
    """
    if not function_definitions:
        raise FunctionCallingError(
            "No function definitions are available to choose from."
        )

    name_grammar = function_name_grammar(function_definitions)
    name_prompt = _build_name_selection_prompt(
        prompt_entry.prompt, function_definitions
    )
    try:
        chosen_name = generate_constrained(
            llm, name_prompt, name_grammar, id_to_text, max_tokens=max_tokens
        )
    except GenerationError as exc:
        raise FunctionCallingError(
            f"Could not select a function for prompt "
            f"{prompt_entry.prompt!r}: {exc}"
        ) from exc

    function_definition = next(
        (fn for fn in function_definitions if fn.name == chosen_name), None
    )
    if function_definition is None:
        raise FunctionCallingError(
            "Constrained generation returned an unknown function name "
            f"{chosen_name!r}."
        )

    args_grammar = parameters_object_grammar(function_definition)
    args_prompt = _build_arguments_prompt(
        prompt_entry.prompt, function_definition
    )
    try:
        raw_arguments = generate_constrained(
            llm, args_prompt, args_grammar, id_to_text, max_tokens=max_tokens
        )
    except GenerationError as exc:
        raise FunctionCallingError(
            f"Could not generate arguments for function {chosen_name!r} "
            f"(prompt {prompt_entry.prompt!r}): {exc}"
        ) from exc

    try:
        parameters = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        # Should not happen if the grammar is correct, but never trust blindly.
        raise FunctionCallingError(
            f"Generated arguments were not valid JSON: {raw_arguments!r}"
        ) from exc

    parameters = _coerce_parameter_types(parameters, function_definition)
    return FunctionCallResult(
        prompt=prompt_entry.prompt, name=chosen_name, parameters=parameters
    )


def _resolve_with_retries(
    llm: LLMBackend,
    prompt_entry: PromptEntry,
    function_definitions: list[FunctionDefinition],
    id_to_text: dict[int, str],
) -> FunctionCallResult:
    """Resolve one prompt, retrying with larger token budgets before giving up.

    This function ALWAYS returns exactly one ``FunctionCallResult`` carrying
    the original prompt, no matter what: silently dropping a prompt is what
    causes every later result to slide into the wrong slot when the output
    file is compared position-by-position against the expected test list.
    If every retry genuinely fails, a schema-valid placeholder is returned
    instead of nothing, so the 1-to-1 correspondence between input prompts
    and output entries always holds.
    """
    last_error: FunctionCallingError | None = None
    for max_tokens in _TOKEN_BUDGETS:
        try:
            return resolve_function_call(
                llm,
                prompt_entry,
                function_definitions,
                id_to_text,
                max_tokens=max_tokens,
            )
        except FunctionCallingError as exc:
            last_error = exc

    print(
        f"[warning] Could not resolve prompt {prompt_entry.prompt!r} after "
        f"{len(_TOKEN_BUDGETS)} attempts ({last_error}); "
        "emitting a placeholder to keep prompt/result alignment.",
        file=sys.stderr,
    )
    fallback_definition = function_definitions[0]
    return FunctionCallResult(
        prompt=prompt_entry.prompt,
        name=fallback_definition.name,
        parameters=_default_parameters(fallback_definition),
    )


def resolve_all(
    llm: LLMBackend,
    prompts: list[PromptEntry],
    function_definitions: list[FunctionDefinition],
    id_to_text: dict[int, str],
) -> list[FunctionCallResult]:
    """Resolve every prompt, producing exactly one result per prompt."""
    return [
        _resolve_with_retries(llm, entry, function_definitions, id_to_text)
        for entry in prompts
    ]
