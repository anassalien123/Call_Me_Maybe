# ABOUTME: Token-by-token constrained generation: at each step only tokens that
# ABOUTME: keep the output on a valid path through a GrammarState are
# ABOUTME: considered.

from __future__ import annotations

from typing import Any, Protocol

from src.grammar import GrammarState

MAX_GENERATED_TOKENS = 200


class LLMBackend(Protocol):
    """The subset of ``Small_LLM_Model`` this module relies on."""

    def get_logits_from_input_ids(
        self, input_ids: list[int]
    ) -> list[float]: ...

    def encode(self, text: str) -> Any: ...  # tensor of shape (1, seq_len)


class GenerationError(RuntimeError):
    """Raised when constrained generation cannot proceed (e.g. exhausted
    grammar)."""


def _prompt_to_ids(llm: LLMBackend, prompt: str) -> list[int]:
    tensor = llm.encode(prompt)
    return [int(x) for x in tensor[0].tolist()]


def generate_constrained(
    llm: LLMBackend,
    prompt: str,
    grammar: GrammarState,
    id_to_text: dict[int, str],
    max_tokens: int = MAX_GENERATED_TOKENS,
) -> str:
    """Generate text token-by-token, only emitting tokens allowed by
    *grammar*.

    At each step, every candidate vocabulary token is checked against the
    current grammar state. Among the tokens that keep the sequence valid, the
    one with the highest logit is greedily selected. Generation stops as soon
    as the grammar reports ``complete`` (a full match has been produced), or
    once no candidate token is a legal continuation.

    Args:
        llm: object exposing ``encode`` and ``get_logits_from_input_ids``.
        prompt: the natural-language prompt to condition generation on.
        grammar: the grammar the generated text must conform to.
        id_to_text: mapping of vocabulary token id to its decoded text.
        max_tokens: safety cap on the number of generation steps.

    Returns:
        The generated text (matching ``grammar``).

    Raises:
        GenerationError: if the grammar cannot be completed within
            ``max_tokens`` steps, or if no valid next token exists.
    """
    input_ids = _prompt_to_ids(llm, prompt)
    state: GrammarState = grammar
    generated_chunks: list[str] = []

    if state.complete:
        return ""

    for _ in range(max_tokens):
        logits = llm.get_logits_from_input_ids(input_ids)

        best_token_id: int | None = None
        best_next_state: GrammarState | None = None
        best_score = float("-inf")

        for token_id, token_text in id_to_text.items():
            if not token_text:
                continue
            next_state = state.advance(token_text)
            if next_state is None:
                continue
            score = (
                logits[token_id] if token_id < len(logits) else float("-inf")
            )
            if score > best_score:
                best_score = score
                best_token_id = token_id
                best_next_state = next_state

        if best_token_id is None or best_next_state is None:
            raise GenerationError(
                "No vocabulary token can legally continue the current "
                "grammar state; constrained generation is stuck."
            )

        input_ids.append(best_token_id)
        generated_chunks.append(id_to_text[best_token_id])
        state = best_next_state

        if state.complete:
            break
    else:
        raise GenerationError(
            f"Grammar was not completed within the {max_tokens}-token "
            "generation budget."
        )

    return "".join(generated_chunks)
