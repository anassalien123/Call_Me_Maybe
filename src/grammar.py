# ABOUTME: Character-level grammar state machines used to constrain token
# ABOUTME: generation. Each grammar exposes step()/complete so a decoder
# ABOUTME: can validate candidate tokens.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.schema import FunctionDefinition

_ESCAPABLE_CHARS = set('"\\/bfnrtu')
_WHITESPACE_CHARS = set(" \t\n\r")


class GrammarState(ABC):
    """A node of a character-level grammar automaton.

    A concrete subclass represents "where we are" while validating a candidate
    string one character at a time. ``step`` either returns the state reached
    after consuming ``char`` or ``None`` if ``char`` is not accepted here.
    """

    @property
    @abstractmethod
    def complete(self) -> bool:
        """Whether the grammar can legally stop generating at this state."""

    @abstractmethod
    def step(self, char: str) -> Optional["GrammarState"]:
        """Consume one character, returning the next state or ``None`` if
        invalid."""

    def advance(self, text: str) -> Optional["GrammarState"]:
        """Consume a (possibly multi-character) string, e.g. one LLM token."""
        state: Optional[GrammarState] = self
        for char in text:
            if state is None:
                return None
            state = state.step(char)
        return state


class LiteralState(GrammarState):
    """Matches one fixed, exact string (e.g. ``{``, ``":``, ``true``)."""

    def __init__(self, text: str, pos: int = 0) -> None:
        self._text = text
        self._pos = pos

    @property
    def complete(self) -> bool:
        return self._pos == len(self._text)

    def step(self, char: str) -> Optional[GrammarState]:
        if self._pos >= len(self._text) or char != self._text[self._pos]:
            return None
        return LiteralState(self._text, self._pos + 1)


class OptionalWhitespaceState(GrammarState):
    """Matches zero or more JSON whitespace characters. Always ``complete``."""

    @property
    def complete(self) -> bool:
        return True

    def step(self, char: str) -> Optional[GrammarState]:
        if char in _WHITESPACE_CHARS:
            return self
        return None


class EnumState(GrammarState):
    """Matches one of a fixed set of literal strings (e.g. function names)."""

    def __init__(self, options: tuple[str, ...], pos: int = 0) -> None:
        self._options = options
        self._pos = pos

    @property
    def complete(self) -> bool:
        return any(len(opt) == self._pos for opt in self._options)

    def step(self, char: str) -> Optional[GrammarState]:
        remaining = tuple(
            opt
            for opt in self._options
            if len(opt) > self._pos and opt[self._pos] == char
        )
        if not remaining:
            return None
        return EnumState(remaining, self._pos + 1)

    @property
    def matched_value(self) -> Optional[str]:
        """Return the exact matched option once ``complete`` is (also) true."""
        for opt in self._options:
            if len(opt) == self._pos:
                return opt
        return None


class StringState(GrammarState):
    """Matches a JSON string literal: ``"..."`` with backslash escapes."""

    _NOT_STARTED, _IN_STRING, _ESCAPED, _DONE = range(4)

    def __init__(self, mode: int = _NOT_STARTED) -> None:
        self._mode = mode

    @property
    def complete(self) -> bool:
        return self._mode == self._DONE

    def step(self, char: str) -> Optional[GrammarState]:
        if self._mode == self._NOT_STARTED:
            return StringState(self._IN_STRING) if char == '"' else None
        if self._mode == self._IN_STRING:
            if char == '"':
                return StringState(self._DONE)
            if char == "\\":
                return StringState(self._ESCAPED)
            if ord(char) < 0x20:
                return None  # raw control chars are not allowed in JSON
            return StringState(self._IN_STRING)
        if self._mode == self._ESCAPED:
            return (
                StringState(self._IN_STRING)
                if char in _ESCAPABLE_CHARS
                else None
            )
        return None  # _DONE: nothing may follow directly


class NumberState(GrammarState):
    """Matches a JSON number: ``-?(0|[1-9]\\d*)(\\.\\d+)?([eE][+-]?\\d+)?``."""

    _SIGN, _ZERO, _INT, _DOT, _FRAC, _EXP_SIGN, _EXP = range(7)

    def __init__(self, mode: int = _SIGN, allow_fraction: bool = True) -> None:
        self._mode = mode
        self._allow_fraction = allow_fraction

    @property
    def complete(self) -> bool:
        return self._mode in (self._ZERO, self._INT, self._FRAC, self._EXP)

    def step(self, char: str) -> Optional[GrammarState]:
        digit = char.isdigit()
        if self._mode == self._SIGN:
            if char == "-":
                return _NumberAfterMinus(self._allow_fraction)
            if char == "0":
                return NumberState(self._ZERO, self._allow_fraction)
            if digit:
                return NumberState(self._INT, self._allow_fraction)
            return None
        if self._mode == self._ZERO:
            if self._allow_fraction and char == ".":
                return NumberState(self._DOT, self._allow_fraction)
            if char in "eE":
                return NumberState(self._EXP_SIGN, self._allow_fraction)
            return None
        if self._mode == self._INT:
            if digit:
                return NumberState(self._INT, self._allow_fraction)
            if self._allow_fraction and char == ".":
                return NumberState(self._DOT, self._allow_fraction)
            if char in "eE":
                return NumberState(self._EXP_SIGN, self._allow_fraction)
            return None
        if self._mode == self._DOT:
            return (
                NumberState(self._FRAC, self._allow_fraction)
                if digit
                else None
            )
        if self._mode == self._FRAC:
            if digit:
                return NumberState(self._FRAC, self._allow_fraction)
            if char in "eE":
                return NumberState(self._EXP_SIGN, self._allow_fraction)
            return None
        if self._mode == self._EXP_SIGN:
            if char in "+-":
                return _NumberExpDigitsPending(self._allow_fraction)
            if digit:
                return NumberState(self._EXP, self._allow_fraction)
            return None
        if self._mode == self._EXP:
            return (
                NumberState(self._EXP, self._allow_fraction) if digit else None
            )
        return None


class _NumberAfterMinus(GrammarState):
    def __init__(self, allow_fraction: bool) -> None:
        self._allow_fraction = allow_fraction

    @property
    def complete(self) -> bool:
        return False

    def step(self, char: str) -> Optional[GrammarState]:
        if char == "0":
            return NumberState(NumberState._ZERO, self._allow_fraction)
        if char.isdigit():
            return NumberState(NumberState._INT, self._allow_fraction)
        return None


class _NumberExpDigitsPending(GrammarState):
    """After the sign following 'e'/'E': at least one digit is mandatory."""

    def __init__(self, allow_fraction: bool) -> None:
        self._allow_fraction = allow_fraction

    @property
    def complete(self) -> bool:
        return False

    def step(self, char: str) -> Optional[GrammarState]:
        if char.isdigit():
            return NumberState(NumberState._EXP, self._allow_fraction)
        return None


def integer_grammar() -> GrammarState:
    return NumberState(allow_fraction=False)


def number_grammar() -> GrammarState:
    return NumberState(allow_fraction=True)


def boolean_grammar() -> GrammarState:
    return EnumState(("true", "false"))


def string_grammar() -> GrammarState:
    return StringState()


def value_grammar(json_type: str) -> GrammarState:
    """Return the grammar for a single JSON value of the given schema type."""
    if json_type == "string":
        return string_grammar()
    if json_type == "integer":
        return integer_grammar()
    if json_type == "number":
        return number_grammar()
    if json_type == "boolean":
        return boolean_grammar()
    # Unknown/unsupported types degrade gracefully to "any JSON string".
    return string_grammar()


class SequenceState(GrammarState):
    """Concatenation of grammars, consumed strictly in order.

    Whenever the currently active part refuses a character but is itself
    ``complete``, the sequence transparently advances to the next part before
    retrying - this lets e.g. a ``NumberState`` finish and a following comma
    literal take over without any external bookkeeping.
    """

    def __init__(
        self, parts: tuple[GrammarState, ...], index: int = 0
    ) -> None:
        self._parts = parts
        self._index = index

    @property
    def complete(self) -> bool:
        if self._index >= len(self._parts):
            return True
        # complete if every remaining part (including the current one) can be
        # skipped/finished without consuming further input
        return all(part.complete for part in self._parts[self._index:])

    def step(self, char: str) -> Optional[GrammarState]:
        index = self._index
        while index < len(self._parts):
            part = self._parts[index]
            new_part = part.step(char)
            if new_part is not None:
                new_parts = (
                    self._parts[:index]
                    + (new_part,)
                    + self._parts[index + 1:]
                )
                return SequenceState(new_parts, index)
            if part.complete:
                index += 1
                continue
            return None
        return None

    def value_at(self, position: int) -> GrammarState:
        return self._parts[position]


def sequence(*parts: GrammarState) -> GrammarState:
    return SequenceState(tuple(parts))


def function_name_grammar(
    function_definitions: list[FunctionDefinition],
) -> GrammarState:
    """Grammar matching exactly one of the known function names."""
    names = tuple(fn.name for fn in function_definitions)
    return EnumState(names)


def parameters_object_grammar(
    function_definition: FunctionDefinition,
) -> GrammarState:
    """Grammar matching a JSON object with exactly the parameters of
    *function_definition*.

    Keys are emitted in the order declared in ``functions_definition.json`` and
    whitespace around ``:``/``,`` is optional, e.g. ``{"a": 1, "b": 2}`` or
    ``{"a":1,"b":2}`` are both accepted.
    """
    parts: list[GrammarState] = [LiteralState("{")]
    items = list(function_definition.parameters.items())
    for position, (param_name, param_def) in enumerate(items):
        parts.append(OptionalWhitespaceState())
        parts.append(LiteralState(f'"{param_name}"'))
        parts.append(OptionalWhitespaceState())
        parts.append(LiteralState(":"))
        parts.append(OptionalWhitespaceState())
        parts.append(value_grammar(param_def.type))
        if position != len(items) - 1:
            parts.append(OptionalWhitespaceState())
            parts.append(LiteralState(","))
    parts.append(OptionalWhitespaceState())
    parts.append(LiteralState("}"))
    return sequence(*parts)
