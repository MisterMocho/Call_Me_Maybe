"""
CLI orchestration, prompt assembly, and result post-processing.

This module is the seam between user input on the command line and the
constrained-decoding engine. It is responsible for three things:

    1. Parsing CLI arguments and loading the two JSON input files.
    2. Building the system prompt that lists the available tools and
       the few-shot examples that anchor the model's regex style.
    3. Running the engine over every test prompt, casting the parsed
       parameters to the types declared in the schema, and writing
       the final results to disk.

The few-shot examples deliberately use placeholder names
(``<tool_name>``, ``<param>``) so that the prompt teaches the model
the expected output shape without leaking the real schema. This is
important because the project is meant to generalise to function
sets and prompts that are unknown at development time.
"""


import json
import argparse
from pathlib import Path
from src.data_loader import load_data
from src.engine import LLMEngine
from src.schemas import FunctionDefinition, TestPrompt


def parse_and_load() -> tuple[list[FunctionDefinition],
                              list[TestPrompt], Path]:
    """
    Parse CLI arguments and load the two JSON input files.

    The function defines three command-line options matching the
    layout required by the subject and falls back to the default
    paths under ``data/`` when none are supplied.

    Returns:
        A tuple ``(tools, prompts, output_path)``:

            * ``tools`` is the validated list of available functions.
            * ``prompts`` is the validated list of user queries to run.
            * ``output_path`` is the destination path for the JSON
              results written at the end of :func:`run_llm`.
    """

    parser = argparse.ArgumentParser(description="Call Me Maybe -"
                                     "Function Calling Inference")
    parser.add_argument("--functions_definition", type=str,
                        default="data/input/functions_definition.json")
    parser.add_argument("--input", type=str,
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output", type=str,
                        default="data/output/function_calling_results.json")
    args = parser.parse_args()

    # Converting file locations to path objects
    func_path = Path(args.functions_definition)
    tests_path = Path(args.input)
    output_path = Path(args.output)

    # 2. Loading Data
    print(f"Loading functions from: {func_path}")
    print(f"Loading tests from: {tests_path}")
    tools, prompts = load_data(func_path, tests_path)

    return tools, prompts, output_path


def run_llm(tools: list[FunctionDefinition],
            prompts: list[TestPrompt],
            output_path: Path) -> None:
    """
    Run the engine over every test prompt and write the results.

    The function performs four logical phases:

        1. Build a single shared system prompt listing every
           available tool in compact JSON form, plus a handful of
           critical rules and few-shot examples that anchor the
           regex style and quoting conventions.
        2. For each user prompt, append it to the shared system
           prompt and call the engine to obtain a constrained JSON
           string.
        3. Parse the JSON, look up the chosen function in the
           schema, and cast each parameter to the declared type
           (``number`` -> ``float``, ``integer`` -> ``int``, etc.).
           This guards against the model emitting a numeric value
           wrapped in quotes and similar minor mismatches.
        4. Collect every result into a list and write it to disk
           in the format expected by the grader.

    Failures during JSON parsing are caught per-prompt: a single
    bad generation cannot abort the whole run.

    Args:
        tools: Validated list of available function definitions.
        prompts: Validated list of user prompts to evaluate.
        output_path: Destination path for the results JSON. The
            parent directory is created if it does not already
            exist.
    """

    engine = LLMEngine()
    final_results = []
    # 4. Base instructions
    base_instructions = (
        "Task: Select the EXACT tool that matches the user's intent.\n"
        "Available tools:\n"
    )
    # Formatting tools as native JSON Objects
    for tool in tools:
        clean_tool = {
            "name": tool.name,
            "parameters": {k: v.type for k, v in tool.parameters.items()}
        }
        # MAGIC: separators=(',', ':') removes all useless spaces!
        base_instructions += (json.dumps(clean_tool, separators=(',', ':')) +
                              "\n")
    base_instructions += (
        "\nCRITICAL RULES:\n"
        "1. Output ONLY a valid JSON calling the correct function.\n"
        "2. Preserve exact punctuation. Escape quotes like \\\".\n"
        "3. For regex parameters: ALWAYS use general character class\n"
        'Vowels: [aeiouAEIOU]\n'
        'EXAMPLES (use the tools listed above, NOT these example names):\n'
        "User prompt: Replace all digits in 'abc123' with X\n"
        'Assistant: {"name":"<tool_name>",'
        '"parameters":{"<param>":"abc123","regex":"[0-9]+","replacement":"X"'
        '}}\n\n'
        "User prompt: Substitute the word 'foo' with 'bar' in 'foo and foo'\n"
        'Assistant: {"name":"<tool_name>",'
        '"parameters":{"<param>":"foo and foo","regex":"foo",'
        '"replacement":"bar"}}\n\n'
        'User prompt: Render string: Welcome "{user}" to the team\n'
        'Assistant: {"name": "<tool_name>", "parameters": '
        '{"template": "Welcome \\"{user}\\" to the team"}}\n\n'
    )
    print("\nInitiating Tests...")

    for i, current_test in enumerate(prompts, 1):
        # Passing the prompts already validated above
        instructions = base_instructions
        instructions += f"User prompt: {current_test.prompt}\n"
        instructions += "Assistant: "

        print(
            f"\n[Test {i}/{len(prompts)}] "
            f"Processing: '{current_test.prompt}'"
        )

        # Generates our enforced answer
        answer_txt = engine.generate(instructions, max_tokens=75)

        try:
            # Reads JSON from the model
            llm_response = json.loads(answer_txt)
            func_name = llm_response.get("name", "unknown_function")
            params = llm_response.get("parameters", {})
            # We look for the function schema that the model chose
            tool_def = next((t for t in tools if t.name == func_name), None)
            if tool_def and tool_def.parameters:
                for p_name, p_val in params.items():
                    if p_name in tool_def.parameters:
                        p_type = tool_def.parameters[p_name].type
                        try:
                            # We transform the data into the correct value
                            if p_type == "number":
                                params[p_name] = float(p_val)
                            elif p_type == "integer":
                                params[p_name] = int(p_val)
                            elif p_type == "string":
                                params[p_name] = str(p_val)
                            elif p_type == "boolean":
                                # Safely converts to boolean
                                if isinstance(p_val, str):
                                    params[p_name] = p_val.lower() in (
                                        ['true', '1', 'yes']
                                    )
                                else:
                                    params[p_name] = bool(p_val)
                        except (ValueError, TypeError):
                            pass
            formatted_result = {
                "prompt": current_test.prompt,
                "name": func_name,
                "parameters": params
            }
            final_results.append(formatted_result)
        except json.JSONDecodeError:
            print(" ❌ (Error decoding JSON)")

    # 4. Saves to the file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    print(f"\nAll tests are now concluded! File saved in: {output_path}")
