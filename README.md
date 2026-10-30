*This project has been created as part of the 42 curriculum by <ancheab>.*

# Call Me Maybe

## Description

**Call Me Maybe** turns natural-language requests ("What is the sum of 2 and
3?") into structured, machine-executable function calls
(`{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}`), using a small
local language model (`Qwen/Qwen3-0.6B`).

Small models are unreliable at spontaneously producing valid JSON — asked
directly, they succeed only a fraction of the time. Instead of hoping the
model "gets the format right", this project uses **constrained decoding**:
at every generation step, every token that would break either JSON syntax or
the target schema is masked out (its logit set to `-inf`) *before* a token is
picked. The model can therefore never emit a syntactically or semantically
invalid character — only tokens that keep the output on a legal path through
the schema are ever considered.

## Instructions

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- The `llm_sdk` package (already vendored under `llm_sdk/` in this repository)

### Installation

```bash
make install
# equivalent to: uv sync
```

`uv sync` installs `pydantic` and `numpy` for the project itself, and
`torch`, `transformers`, `huggingface_hub` as declared by the vendored
`llm_sdk` package. The first run will also download the `Qwen/Qwen3-0.6B`
weights and tokenizer files from the Hugging Face Hub.

### Running

```bash
make run
# equivalent to:
# uv run python -m src \
#     --functions_definition data/input/functions_definition.json \
#     --input data/input/function_calling_tests.json \
#     --output data/output/function_calling_results.json
```

All three flags are optional and default to the paths above. A custom model
can be selected with `--model <hf-repo-id>` (it must be a causal LM with a
byte-level BPE tokenizer, compatible with `llm_sdk.Small_LLM_Model`).

Other Makefile targets:

- `make debug` — run the program under `pdb`
- `make test` — run the (non-graded) unit test suite with `pytest`
- `make lint` / `make lint-strict` — run `flake8` and `mypy`
- `make clean` — remove caches and the generated `data/output/` directory

## Algorithm Explanation

Generation happens in two constrained phases per prompt, both driven by the
same generic engine (`src/decoding.py`):

1. **Function selection.** The model is asked, in natural language, to pick
   one function name from the catalog in `functions_definition.json`.
   Generation is constrained by an `EnumState` grammar (`src/grammar.py`)
   built from the exact list of function names: at each step only tokens
   that are a valid continuation of *at least one* remaining candidate name
   are allowed. As soon as the accumulated text exactly matches one full
   name, generation stops. The model cannot select a nonexistent function or
   produce a name with a typo — the grammar physically prevents it.

2. **Argument extraction.** Once a function is fixed, its parameter schema
   (name, order, and JSON type of each argument) is compiled into a
   `SequenceState` grammar: a concatenation of literal tokens (`{`, `"a"`,
   `:`, `,`, `}`) and per-type value grammars (`StringState` for strings,
   `NumberState` for numbers/integers, an enum of `true`/`false` for
   booleans). The model is prompted to fill in the arguments for that
   specific function, and generation is constrained by this object grammar.
   The result is guaranteed to be a JSON object with exactly the required
   keys, in the required order, each holding a value of the required type.

At every generation step (`generate_constrained`):

1. The current input ids are sent to `llm.get_logits_from_input_ids`.
2. Every vocabulary token is decoded once, up front (`src/vocab.py`, using
   the standard byte-level BPE `bytes_to_unicode` mapping) into the literal
   text it represents.
3. For each token, the grammar is asked "if I appended this token's text,
   would I still be on a legal path?" (`GrammarState.advance`). Tokens that
   would break the grammar are skipped entirely — equivalent to setting
   their logit to `-inf`.
4. Among the remaining legal tokens, the one with the highest logit is
   picked (greedy decoding), which is the point where the model's actual
   "opinion" still gets to matter.
5. Generation stops as soon as the grammar reports the sequence is
   `complete` (a full match).

Because invalid tokens are never candidates in the first place, the output
is **always** syntactically valid JSON that satisfies the schema — this is
enforced structurally, not by hoping the model behaves or by
post-hoc validation/repair.

## Design Decisions

- **Hand-rolled character-level grammar instead of a full JSON-Schema
  library.** The set of shapes this project needs (an enum of function
  names, and a fixed-order object of typed scalar values) is small and
  well-defined, so a small set of composable `GrammarState` classes
  (`LiteralState`, `EnumState`, `StringState`, `NumberState`,
  `SequenceState`) is easier to reason about, test and keep dependency-free
  than integrating a general JSON-Schema-to-automaton library — and it
  keeps the "why is this token valid or not" logic fully transparent.
- **Two separate constrained generations (name, then arguments)** rather
  than one grammar for the whole `{"name": ..., "parameters": {...}}`
  object. The parameter schema depends on *which* function is chosen, which
  a single static grammar cannot express; splitting the problem into
  "choose from an enum" followed by "fill in this specific schema" keeps
  each grammar static and simple.
- **`prompt` is not generated by the model.** The output schema requires
  echoing back the original natural-language prompt verbatim; since we
  already have that string, it is copied into the result directly instead
  of wastefully re-generating it (and risking the model paraphrasing it).
- **Pydantic models for all inputs/outputs** (`src/schema.py`), per the
  project's additional requirements. Both input files are validated on
  load, and the output is only ever assembled through `FunctionCallResult`
  before being serialized to JSON.
- **Unsupported/unknown parameter types degrade to a string grammar**
  rather than crashing, matching the "handle errors gracefully" requirement,
  since the mandatory examples only ever use `string`, `number`
  and (implicitly) `boolean`.
- **Parameter values are coerced to their declared JSON type after parsing.**
  The grammar only guarantees a value is a *syntactically* valid JSON number
  (`3` and `3.0` are both legal), not which Python type `json.loads` produces
  for it. `_coerce_parameter_types` in `src/function_caller.py` casts every
  generated value to `float`/`int`/`bool` according to
  `functions_definition.json`, without touching the underlying value, so a
  `number`-typed argument is always serialized as `3.0`, never `3`.
- **Every input prompt always produces exactly one output entry, in order.**
  `resolve_all` never silently drops a prompt whose generation fails: it
  retries with progressively larger token budgets
  (`src/function_caller.py::_TOKEN_BUDGETS`), and only falls back to a
  schema-valid placeholder as an absolute last resort. Dropping an entry
  would shift every later result into the wrong slot when the output file is
  compared position-by-position against an expected list — silently eating
  one failure used to cause a cascade of "wrong" results after it.

## Performance Analysis

- **Accuracy:** every produced entry is 100% schema-valid JSON by
  construction — this is not a probabilistic property, it is guaranteed by
  the grammar. Whether the *content* is correct (the right function, the
  right values) still depends on the underlying model's ability to
  understand the prompt; that part is standard LLM prompting, unconstrained
  in what it says other than "shaped like the schema".
- **Speed:** a naive implementation of `generate_constrained` scans the
  entire vocabulary at every generation step. For a small model and short
  outputs (function names and a handful of scalar arguments), this stays
  well within the "a few minutes for the whole test set" requirement on
  commodity hardware; the bonus part explicitly calls out caching/batching
  as further optimizations that were intentionally left out of the
  mandatory implementation for clarity.
- **Reliability:** malformed or missing input files, an unreachable model,
  or a grammar that cannot be satisfied within the token budget all raise a
  specific, caught exception with a clear message (`ProjectIOError`,
  `GenerationError`, `FunctionCallingError`) instead of crashing the
  program; a single unresolved prompt is skipped with a warning rather than
  aborting the whole run.

## Challenges Faced

- **Turning a JSON Schema into a per-character automaton** that supports
  optional whitespace and fixed key ordering without becoming an ad-hoc
  regex engine — solved by making `SequenceState` transparently skip to the
  next part whenever the current part reports itself `complete()` and
  refuses the next character, which lets whitespace and terminated
  values "get out of the way" automatically.
- **Decoding byte-level BPE tokens correctly.** Tokens are stored using the
  GPT-2 byte-to-unicode trick (e.g. `Ġ` represents a leading space); getting
  the reverse mapping wrong silently produces garbage text that would
  quietly fail every grammar check. This was verified independently against
  known token/text pairs.
- **Avoiding private `llm_sdk` internals**, as explicitly forbidden by the
  subject — the implementation only ever calls `encode`,
  `get_logits_from_input_ids`, and `get_path_to_vocab_file`.

## Testing Strategy

Test programs (not graded, per the subject) live under `tests/` and are run
with `pytest`:

- `tests/test_grammar.py` — exercises every grammar primitive directly
  (valid/invalid strings, numbers, booleans, enums, and full parameter
  objects), including edge cases like leading zeros, missing/extra keys, and
  wrong value types.
- `tests/test_vocab.py` — verifies byte-level BPE decoding against known
  token/text pairs.
- `tests/test_decoding.py` — drives the full constrained-generation loop
  against a fake LLM backend that deterministically "wants" to produce a
  known target string, checking both the success path and the "grammar
  unreachable" failure path.
- `tests/test_function_caller.py` — exercises the two-phase orchestration
  (name selection, then argument extraction) end-to-end against a fake LLM,
  including error propagation and `resolve_all`'s "skip failing prompts"
  behaviour.
- `tests/test_io_utils.py` — file loading/saving, including missing files,
  invalid JSON, and wrong top-level shape.

None of the tests download or run the real `Qwen/Qwen3-0.6B` model, so the
suite runs quickly and offline; a fake LLM backend stands in wherever model
behaviour would otherwise be required.

## Example Usage

```bash
$ make run
Wrote 11 result(s) to data/output/function_calling_results.json

$ cat data/output/function_calling_results.json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2, "b": 3}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  }
]
```

## Checking Your Implementation with Moulinette

You can use the provided Moulinette to generate a private test set, run your
implementation against it, and automatically grade the generated function
calls.

### 1. Install Dependencies

From the project root, synchronize the project dependencies:

```bash
make install
```

### 2. Generate the Private Exercise Set

Navigate to the `moulinette` directory:

```bash
cd moulinette
```

Generate the private exercise files:

```bash
uv run python -m moulinette prepare_exercises --set private
```

This generates the required test and correction files inside the
`moulinette/data` directory:

```
moulinette/data/
├── input/
│   ├── functions_definition.json
│   └── function_calling_tests.json
└── correction/
    └── function_calling_corrections.json
```

### 3. Run Your Implementation

Return to the project root:

```bash
cd ..
```

Run the project using the generated private test set:

```bash
uv run python -m src \
    --input moulinette/data/input/function_calling_tests.json \
    --functions_definition moulinette/data/input/functions_definition.json \
    --output data/output/function_calls.json
```

The generated function calls will be saved to:

```
data/output/function_calls.json
```

### 4. Grade Your Results

Navigate back to the `moulinette` directory:

```bash
cd moulinette
```

Run the private test grader:

```bash
uv run python -m moulinette grade_student_answers \
    --set private \
    --student_answer_path ../data/output/function_calls.json
```

The Moulinette will run the generated answers against the private test set
and report whether each function call is valid.

> **Note:** The private test set is generated locally by the Moulinette.
> Make sure to run `prepare_exercises` before grading if the required
> private exercise files have not yet been generated.

## Resources

- [OpenAI: Function calling](https://platform.openai.com/docs/guides/function-calling)
- [Hugging Face: Byte-Pair Encoding tokenization](https://huggingface.co/learn/nlp-course/chapter6/5)
- [Outlines: structured generation with constrained decoding](https://github.com/dottxt-ai/outlines)
- [Guidance: constrained/grammar-guided generation](https://github.com/guidance-ai/guidance)
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259)
- Qwen3 model card: [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B)
