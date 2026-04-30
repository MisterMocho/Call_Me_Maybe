*This project has been created as part of the 42 curriculum by luida-cu.*

# Call Me Maybe — Function Calling in LLMs

## 📖 Description

Large Language Models are powerful at processing natural language but unreliable at producing strictly formatted, machine-readable output. This project implements a **Function Calling Agent** built on top of a small 0.6B parameter Qwen model that guarantees syntactically valid JSON responses adhering to predefined function schemas.

Rather than relying on prompt engineering and hoping the model behaves, the engine implements **Constrained Decoding**: it intercepts the raw probability scores (logits) of the model token-by-token during inference and applies masks that mathematically forbid any token that would violate the JSON grammar. The result is 100% valid JSON output, by construction.

## 🚀 Instructions

### Prerequisites
- Python 3.10+
- `uv` package manager

### Installation
```bash
make install
```

### Execution
Default paths (`data/input/` and `data/output/`):
```bash
make run
```

Custom paths:
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

## 🧠 Algorithm

The core of the engine is a token-by-token interception loop that combines four cooperating mechanisms.

### 1. Prefix Injection
Before generation begins, the engine pre-encodes the deterministic JSON prefix `{"name":"` and appends those tokens directly into the model's input context. The model never has to "discover" how to start a JSON object — it begins generation already inside the first string value, which is the function name. This eliminates several inference calls per query.

### 2. Logit Masking (Schema Enforcement)
Two precomputed masks are added to the raw logits at each generation step:
- **`mask_json`** — permits all characters that may legitimately appear inside a function-calling JSON (alphanumerics, common punctuation, brackets, regex metacharacters).
- **`mask_strict_keys`** — a tighter mask applied immediately after `{` or `,`, where only a JSON key can legally follow.

A lightweight finite-state machine selects which mask to apply based on the last structural character seen outside any string.

### 3. Fast-Forwarding
At positions where the next token is structurally determined (e.g., the `:` after `"name"`, the opening `{` after `"parameters":`), the engine bypasses the model entirely and injects the token directly. The model is only consulted when it has genuine choice — for the function name itself and for parameter values.

### 4. Brace Counter (Deterministic Halting)
The engine tracks `open_brackets` while ignoring braces that appear inside string values. When the counter reaches zero, generation halts immediately — preventing the model from emitting any conversational prose after the JSON closes.

### State Tracking
Three pieces of state persist across the entire generation and across token boundaries:
- `inside_string` — toggles on every unescaped `"`.
- `prev_char` — the last character emitted, used to detect escape sequences (e.g., `\"`) even when the BPE tokenizer splits `\` and `"` into separate tokens.
- `last_structural_char` — the most recent structural JSON character (`{`, `}`, `:`, `,`) seen outside a string.

These three values are what allow the engine to correctly handle adversarial inputs containing braces, commas, quotes, and backslashes inside parameter values.

## 🏗️ Design Decisions

**Separation of concerns.** The architecture is partitioned into focused modules:
- `data_loader.py` — file reading and validation
- `schemas.py` — Pydantic models for function definitions and prompts
- `vocab_loader.py` — BPE vocabulary loading and lookup
- `engine.py` — inference, masking, and decoding logic
- `parseandrun.py` — CLI orchestration, prompt assembly, and output formatting
- `__main__.py` — entry point

**Strict compliance with the subject.** No direct imports of `pytorch` or `transformers`. The decoding logic uses only `numpy` for numerical operations and standard Python for everything else. Tensor outputs from the SDK are sanitized via `.tolist()` before being touched.

**Pydantic validation.** Function definitions and test prompts are validated at load time, ensuring data integrity before any inference begins.

**Few-shot prompting without leaking the schema.** The system prompt includes two regex-style examples that use generic placeholder function names (not the actual schema entries). This teaches the model the correct regex style (`[0-9]+` rather than `34|233`, literal `cat` rather than `cat.*cat`) without revealing which functions exist.

**Pydantic-driven type casting.** After JSON parsing, parameter values are cast to the type declared in the schema (`number` → `float`, `integer` → `int`, etc.), guarding against the model emitting `"42"` where `42` was expected.

## 📊 Performance

- **Accuracy:** 11/11 on the test suite.
- **Reliability:** 100% valid JSON output guaranteed by the masking and brace-counting halting condition.
- **Speed:** Generation completes in seconds per query when running on a GPU. Fast-forwarding eliminates roughly half the inference calls per generation; the prefix injection alone removes the first 4-6 model calls.

## 🧗 Challenges & Resolutions

The system reached 11/11 through several iterations. The most instructive bugs are summarized below.

**1. The Tensor Trap.** The subject documented `encode` as returning `List[int]`, but the SDK actually returns a 2D PyTorch Tensor. Resolved by indexing `[0]` and calling `.tolist()` — no `import torch` required.

**2. Hidden SDK dependencies.** The provided `llm_sdk` silently requires `accelerate` to map model weights on cluster machines. Adding `accelerate` to `pyproject.toml` makes the SDK runnable; it is never imported from `src/`.

**3. The comma-inside-string bug.** Initial fast-forwarding injected a `"` after every `,`, which corrupted SQL strings like `INSERT INTO logs VALUES (1, 2, 3)`. Resolved by gating the fast-forward on `inside_string`.

**4. Braces inside string values.** Templates like `Hello {user}` caused the brace counter to incorrectly increment, preventing the JSON from closing. Resolved by tracking `inside_string` and only counting braces outside strings.

**5. Cross-token escape detection.** The BPE tokenizer can split `\"` into separate tokens (`\` and `"`), or fuse them into a single token, depending on float precision differences between CPU and GPU runs. The same code produced different output on different machines. Resolved by persisting `prev_char` across token boundaries rather than resetting it inside each decoded token.

**6. Regex hallucination on a 0.6B model.** Without guidance, the model emitted literal patterns like `34|233` instead of `[0-9]+`. Resolved with a minimal few-shot block in the prompt that shows two regex examples using placeholder function names — enough to anchor the right pattern style without revealing the schema.

## 🧪 Testing Strategy

- **Iterative prompt refinement.** The instructions block was tightened across multiple runs, with each rule justified by a failure mode actually observed during evaluation.
- **Robustness.** All JSON parsing is wrapped in `try/except`; a generation failure for one query never crashes the run.
- **Cross-machine validation.** The bug at challenge #5 above was only discovered by running the same code on both a personal laptop (GPU) and a school workstation (CPU fallback). The fix was confirmed by re-running on both.

## 💡 Example Usage

**Input:**
> "Replace all numbers in 'Hello 34 I'm 233 years old' with NUMBERS"

**Output JSON:**
```json
{
    "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
        "source_string": "Hello 34 I'm 233 years old",
        "regex": "[0-9]+",
        "replacement": "NUMBERS"
    }
}
```

## 📚 Resources

- [Understanding Large Language Models](https://en.wikipedia.org/wiki/Large_language_model)
- [Constrained Decoding in Modern AI](https://huggingface.co/docs/transformers/main/en/generation_strategies)

**AI usage.** AI assistants were used as a pair-programming partner during development. Concretely:
- Discussing trade-offs around the few-shot prompt design (specifically, the line between teaching regex style and leaking the schema).
- Reviewing the code after the prefix injection refactor.

The constrained-decoding algorithm itself, the brace-counter halting condition, the FSM mask selection, and all production code in `src/` were designed and written by hand.
