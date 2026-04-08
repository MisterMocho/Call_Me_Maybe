from pathlib import Path
from src.data_loader import load_data
from src.engine import LLMEngine


def main() -> None:
    input_dir = Path("data/input")

    # 1. Carregar os Dados (Lógica escondida no data_loader.py)
    tools, prompts = load_data(input_dir)
    print(
        f"Ready to proccess {len(prompts)}"
        f"tests using {len(tools)} functions."
    )
    # 2. Iniciar a Máquina (Lógica escondida no engine.py)
    engine = LLMEngine()

    # 3. Testar
    ids_json = engine.get_json_symbols_ids()
    print(f"\nIDs para chaveta aberta: {ids_json['{']}")


if __name__ == "__main__":
    main()
