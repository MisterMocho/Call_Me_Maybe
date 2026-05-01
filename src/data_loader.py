"""
Load and validate the two JSON input files required by the engine.

Both files are validated against Pydantic schemas at load time so that
malformed input fails fast with a clear error message, rather than
producing silent corruption deeper in the pipeline.
"""


import sys
import json
from pathlib import Path
from pydantic import TypeAdapter
from src.schemas import FunctionDefinition, TestPrompt


def load_data(func_defs_path: Path,
              tests_path: Path) -> tuple[list[FunctionDefinition],
                                         list[TestPrompt]]:
    """
    Read and validate the function definitions and test prompts files.

    Each file is read as JSON and parsed through Pydantic's
    :class:`TypeAdapter`, which enforces the schemas declared in
    :mod:`src.schemas`. On any read or validation error, the program
    exits with status code 1 after printing a diagnostic; the caller
    is therefore guaranteed to receive well-formed lists or never
    return at all.

    Args:
        func_defs_path: Filesystem path to the functions definition
            JSON. Expected to contain a list of objects matching
            :class:`FunctionDefinition`.
        tests_path: Filesystem path to the test prompts JSON. Expected
            to contain a list of objects matching :class:`TestPrompt`.

    Returns:
        A tuple ``(tools, prompts)`` where ``tools`` is the validated
        list of function definitions available to the engine and
        ``prompts`` is the validated list of user queries to run.

    Raises:
        SystemExit: If either file cannot be opened, parsed as JSON,
            or validated against its schema.
    """

    try:
        with open(func_defs_path, "r", encoding="utf-8") as f:
            raw_defs = json.load(f)
        tools = TypeAdapter(list[FunctionDefinition]).validate_python(raw_defs)
        print(f"Success: {len(tools)} functions loaded and validated.")
    except Exception as e:
        print(f"Definition Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(tests_path, "r", encoding="utf-8") as f:
            raw_prompts = json.load(f)
        prompts = TypeAdapter(list[TestPrompt]).validate_python(raw_prompts)
        print(f"Success {len(prompts)} test prompts loaded")
        return tools, prompts
    except Exception as e:
        print(f"Error on the tests: {e}", file=sys.stderr)
        sys.exit(1)
