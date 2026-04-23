from src.parseandrun import parse_and_load, run_llm


def main() -> None:
    # 1. Argument config as asked by the subject
    tools, prompts, output = parse_and_load()
    # 2. Run LLM with the correct file paths, prompts and output file path
    run_llm(tools, prompts, output)


if __name__ == "__main__":
    main()
