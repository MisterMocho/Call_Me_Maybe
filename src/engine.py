import math
from pathlib import Path
from typing import Any
from llm_sdk import Small_LLM_Model
from src.vocab_loader import VocabLoader


class LLMEngine:
    def __init__(self) -> None:
        print("Initializing LLM engine...")
        self.model: Any = Small_LLM_Model()
        vocab_path_str: str = str(self.model.get_path_to_vocab_file())
        self.vocab = VocabLoader(Path(vocab_path_str))
        print("LLM Engine and Vocab ready!")

    def get_json_symbols_ids(self) -> dict[str, set[int]]:
        symbols = ['{', '}', ':', ',', '"']
        return {sym: self.vocab.find_tokens_for_char(sym) for sym in symbols}

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        print("\n--- Intercepting Model ---")
        # Transforms the prompt text in ID's the llm understands
        input_tensor = self.model.encode(prompt)
        input_ids = input_tensor[0].tolist()
        # Finds which IDs are allowed to use
        allowed_ids_to_start = self.get_json_symbols_ids()['{']
        generated_ids: list[int] = []
        open_brackets = 0
        for step in range(max_tokens):
            # Asks for next Token logits
            logits = self.model.get_logits_from_input_ids(input_ids)
            if step == 0:
                for i in range(len(logits)):
                    if i not in allowed_ids_to_start:
                        logits[i] = -math.inf
            next_token_id = logits.index(max(logits))
            if next_token_id == 151645:
                break
            generated_ids.append(next_token_id)
            input_ids.append(next_token_id)
            generated_word = self.model.decode([next_token_id])
            print(generated_word, end="", flush=True)
            if '{' in generated_word:
                open_brackets += 1
            if '}' in generated_word:
                open_brackets -= 1
            if step > 0 and open_brackets == 0:
                break
        print("\n\n---Generation Completed---")
        return str(self.model.decode(generated_ids))
