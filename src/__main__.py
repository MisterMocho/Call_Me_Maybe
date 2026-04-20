import json
import argparse
from pathlib import Path
from src.data_loader import load_data
from src.engine import LLMEngine


def main() -> None:
    # 1. Argument config as asked by the subject
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
    # 3. Starting the Engine
    engine = LLMEngine()
    final_results = []
    # 4. Base instructions
    base_instructions = (
        "You are an expert AI assistant. Your task is to call"
        " a function to help the user.\n"
    )
    base_instructions += "You have access to the following tools:\n\n"

    for tool in tools:
        base_instructions += f"Function Name: {tool.name}\n"
        base_instructions += f"Description: {tool.description}\n"
        base_instructions += "Parameters:\n"
        if tool.parameters:
            for param_name, param_def in tool.parameters.items():
                base_instructions += (
                    f"  - {param_name} (type: {param_def.type})\n"
                )
        else:
            base_instructions += "  None\n"
        base_instructions += "----------\n"

    base_instructions += (
        "\nBased on the user's query, output ONLY a JSON "
        "object calling the correct function.\n"
    )
    base_instructions += "The JSON must strictly follow this format:\n"
    base_instructions += (
        '{"name": "function_name", "parameters": {"param1": "value"}}\n\n'
    )
    print("\nInitiating Tests...")

    for i, current_test in enumerate(prompts, 1):
        # Passing the prompts already validated above
        instructions = base_instructions
        instructions += f"User prompt: {current_test.prompt}\n"
        instructions += "Assistant: "

        print(
            f"\n[Test {i}/{len(prompts)}] "
            f"Processing: '{current_test.prompt}'", end="", flush=True
        )

        # Gerar resposta forçada
        answer_txt = engine.generate(instructions, max_tokens=150)

        try:
            # Lemos o JSON do modelo
            llm_response = json.loads(answer_txt)

            # Formatamos estritamente como o Subject pede
            formatted_result = {
                "prompt": current_test.prompt,
                "name": llm_response.get("name", "unknown_function"),
                "parameters": llm_response.get("parameters", {})
            }
            final_results.append(formatted_result)
            print(" ✅")

        except json.JSONDecodeError:
            print(" ❌ (Error decoding JSON)")

    # 4. Guardar no ficheiro exigido pelo subject
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # A escola geralmente avalia estes JSONs formatados de forma legível
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"\nAll tests are now concluded! File saved in: {output_path}")


if __name__ == "__main__":
    main()
