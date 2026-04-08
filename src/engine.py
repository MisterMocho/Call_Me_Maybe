from pathlib import Path
from typing import Any
from llm_sdk import Small_LLM_Model
from vocab_loader import VocabLoader


class LLMEngine:
    def __init__(self) -> None:
        print("Initializing LLM engine...")
        self.model: Any = Small_LLM_Model()
        vocab_path_str: str = str(self.model.get_path_to_vocabulary_json())
        self.vocab = VocabLoader(Path(vocab_path_str))
        print("LLM Engine and Vocab ready!")

    def get_json_symbols_ids(self) -> dict[str, set[int]]:
        symbols = ['{', '}', ':', ',', '"']
        return {sym: self.vocab.find_tokens_for_char(sym) for sym in symbols}
