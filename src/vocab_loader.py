import json
from pathlib import Path


class VocabLoader:
    def __init__(self, vocab_path: Path) -> None:
        self.vocab_path = vocab_path
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._load_vocab()

    def _load_vocab(self) -> None:
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {value: k for k, value in self.token_to_id.items()}

    def get_token_id(self, token_str: str) -> int | None:
        return self.token_to_id.get(token_str)

    def find_tokens_for_char(self, char: str) -> set[int]:
        valid_ids: set[int] = set()
        for token_str, token_id in self.token_to_id.items():
            if token_str == char or token_str == f"Ġ{char}":
                valid_ids.add(token_id)
        return valid_ids
