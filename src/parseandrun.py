import json
import argparse
from pathlib import Path
from src.data_loader import load_data
from src.engine import LLMEngine
from src.schemas import FunctionDefinition, TestPrompt


def parse_and_load() -> tuple[list[FunctionDefinition],
                              list[TestPrompt], Path]:
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
    engine = LLMEngine()
    final_results = []
    # 4. Base instructions
    base_instructions = (
        "Task: Select the EXACT tool that matches the user's intent.\n"
        "Available tools:\n"
    )
    # Formatar as tools como objetos JSON nativos (o modelo ADORA isto)
    for tool in tools:
        clean_tool = {
            "name": tool.name,
            "description": tool.description,
            "parameters": {k: v.type for k, v in tool.parameters.items()}
        }
        # A MAGIA AQUI: separators=(',', ':') remove todos os espaços inúteis!
        base_instructions += (json.dumps(clean_tool, separators=(',', ':')) +
                              "\n")
    base_instructions += (
        "\nCRITICAL RULES:\n"
        "1. Output ONLY a valid JSON calling the correct function.\n"
        "2. Preserve exact punctuation. Escape quotes like \\\".\n"
        "3. For regex parameters: ALWAYS use general character class"
        " Numbers: '[0-9]+'. Vowels: '[aeiouAEIOU]'\n"
        " and exact literal replacements (e.g., '*' not '***').\n"
        " Literal words are OK only when replacing exact strings like 'cat'"
        " for 'dog'"
        "4. COPY string parameters VERBATIM from the user prompt.\n"
        'User prompt: Render string: Welcome "{user}" to the team\n'
        'Assistant: {"name": "fn_render", "parameters": '
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

        # Gerar resposta forçada
        answer_txt = engine.generate(instructions, max_tokens=75)

        try:
            # Lemos o JSON do modelo
            llm_response = json.loads(answer_txt)
            func_name = llm_response.get("name", "unknown_function")
            params = llm_response.get("parameters", {})
            # Vamos procurar o esquema da função que o LLM escolheu
            tool_def = next((t for t in tools if t.name == func_name), None)
            if tool_def and tool_def.parameters:
                for p_name, p_val in params.items():
                    if p_name in tool_def.parameters:
                        p_type = tool_def.parameters[p_name].type
                        try:
                            # Forçar o tipo exato exigido pela escola
                            if p_type == "number":
                                params[p_name] = float(p_val)
                            elif p_type == "integer":
                                params[p_name] = int(p_val)
                            elif p_type == "string":
                                params[p_name] = str(p_val)
                            elif p_type == "boolean":
                                # Converte de forma segura para booleanos
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

    # 4. Guardar no ficheiro exigido pelo subject
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # A escola geralmente avalia estes JSONs formatados de forma legível
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    print(f"\nAll tests are now concluded! File saved in: {output_path}")
