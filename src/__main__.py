"""
Entry point for the Call Me Maybe function-calling agent.

Wires together argument parsing, data loading, and the inference loop.
Run as a module from the project root:

    uv run python -m src \\
        --functions_definition data/input/functions_definition.json \\
        --input data/input/function_calling_tests.json \\
        --output data/output/function_calling_results.json
"""


from src.parseandrun import parse_and_load, run_llm


def main() -> None:
    """Parse CLI arguments, load inputs, and run the LLM over all prompts."""
    # 1. Argument config as asked by the subject
    tools, prompts, output = parse_and_load()
    # 2. Run LLM with the correct file paths, prompts and output file path
    run_llm(tools, prompts, output)


if __name__ == "__main__":
    main()
