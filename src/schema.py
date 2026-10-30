# ABOUTME: Pydantic models describing the input/output schemas used by
# ABOUTME: the project. Covers function definitions, test prompts and the
# ABOUTME: final generation result.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# JSON schema primitive types we know how to constrain generation for.
SUPPORTED_TYPES = ("string", "number", "integer", "boolean")


class ParameterDefinition(BaseModel):
    """Describes a single argument of a callable function."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        ...,
        description=(
            "JSON type of the parameter (string, number, integer, boolean)."
        ),
    )

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        if value not in SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported parameter type '{value}'. "
                f"Supported types: {SUPPORTED_TYPES}"
            )
        return value


class ReturnDefinition(BaseModel):
    """Describes the return type of a callable function (informational
    only)."""

    model_config = ConfigDict(extra="allow")

    type: str


class FunctionDefinition(BaseModel):
    """A single entry of ``functions_definition.json``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterDefinition] = Field(default_factory=dict)
    returns: ReturnDefinition | None = None


class PromptEntry(BaseModel):
    """A single entry of ``function_calling_tests.json``."""

    model_config = ConfigDict(extra="forbid")

    prompt: str


class FunctionCallResult(BaseModel):
    """A single entry written to ``function_calling_results.json``."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]
