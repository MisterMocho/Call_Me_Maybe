import json
import argparse
from pathlib import Path
from src.data_loader import load_data
from src.engine import LLMEngine


def main() -> None:
    # 1. Configurar os argumentos de linha de comandos exigidos pelo subject
    parser = argparse.ArgumentParser(description="Call Me Maybe -"
                                     "Function Calling Inference")
    parser.add_argument("--functions_definition", type=str,
                        default="data/input/functions_definition.json")
    parser.add_argument("--input", type=str,
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output", type=str,
                        default="data/output/function_calling_results.json")
    args = parser.parse_args()

    # Converter caminhos para Path objects
    func_path = Path(args.functions_definition)
    tests_path = Path(args.input)
    output_path = Path(args.output)

    # 2. Carregar os Dados
    print(f"Loading functions from: {func_path}")
    print(f"Loading tests from: {tests_path}")
    tools, prompts = load_data(func_path.parent)
    # 3. Iniciar o Motor
    engine = LLMEngine()
    final_results = []

    print("\n🚀 A INICIAR BATERIA DE TESTES...")

    for i, current_test in enumerate(prompts, 1):
        # O mesmo prompt de sistema que já validámos que funciona
        instrucoes = (
            "You are an expert AI assistant. Your task is to call"
            " a function to help the user.\n"
        )
        instrucoes += "You have access to the following tools:\n\n"

        for tool in tools:
            instrucoes += f"Function Name: {tool.name}\n"
            instrucoes += f"Description: {tool.description}\n"
            instrucoes += "Parameters:\n"
            if tool.parameters:
                for param_name, param_def in tool.parameters.items():
                    instrucoes += (
                        f"  - {param_name} (type: {param_def.type})\n"
                    )
            else:
                instrucoes += "  None\n"
            instrucoes += "----------\n"

        instrucoes += (
            "\nBased on the user's query, output ONLY a JSON "
            "object calling the correct function.\n"
        )
        instrucoes += "The JSON must strictly follow this format:\n"
        instrucoes += (
            '{"name": "function_name", "parameters": {"param1": "value"}}\n\n'
        )
        instrucoes += f"User prompt: {current_test.prompt}\n"
        instrucoes += "Assistant: "

        print(
            f"\n[Teste {i}/{len(prompts)}] "
            f"A processar: '{current_test.prompt}'", end="", flush=True
        )

        # Gerar resposta forçada
        answer_txt = engine.generate(instrucoes, max_tokens=150)

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
            print(" ❌ (Erro ao descodificar JSON)")

    # 4. Guardar no ficheiro exigido pelo subject
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # A escola geralmente avalia estes JSONs formatados de forma legível
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 Bateria concluída! Ficheiro guardado em: {output_path}")


if __name__ == "__main__":
    main()
