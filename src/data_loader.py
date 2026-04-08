import sys
import json
from pathlib import Path
from pydantic import TypeAdapter
from schemas import FunctionDefinition, TestPrompt


def load_data(input_dir: Path) -> tuple[list[FunctionDefinition],
                                        list[TestPrompt]]:
    func_defs_path = input_dir / "functions_definition.json"
    tests_path = input_dir / "function_calling_tests.json"

    try:
        with open(func_defs_path, "r", encoding="utf-8") as f:
            raw_defs = json.load(f)
        tools = TypeAdapter(list[FunctionDefinition]).validate_python(raw_defs)
        print(f"Success: {len(tools)} functions loaded and validated.")
    except Exception as e:
        print(f"Definition Error: {e}")
        sys.exit(1)

    try:
        with open(tests_path, "r", encoding="utf-8") as f:
            raw_prompts = json.load(f)
        prompts = TypeAdapter(list[TestPrompt]).validate_python(raw_prompts)
        print(f"Success {len(prompts)} test prompts loaded")
        return tools, prompts
    except Exception as e:
        print(f"Error on the tests: {e}")
        sys.exit(1)
