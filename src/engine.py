import numpy as np
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
        symbols = ['{', '"', ':']
        json_symbols = {sym: self.vocab.find_tokens_for_char(sym)
                        for sym in symbols}
        self.allowed_step_0 = set()
        for t_id in json_symbols['{']:
            token_str = self.vocab.id_to_token.get(t_id, "")
            if token_str.replace("Ġ", "").strip() == '{':
                self.allowed_step_0.add(t_id)
        self.allowed_step_1 = set()
        for t_id in json_symbols['"']:
            token_str = self.vocab.id_to_token.get(t_id, "")
            if token_str.replace("Ġ", "").strip().startswith('"'):
                self.allowed_step_1.add(t_id)
        self.t_quote = next(iter(json_symbols['"']))
        self.t_colon = next(iter(json_symbols[':']))
        self.t_open_brace = next(iter(json_symbols['{']))
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEF"
                            "GHIJKLMNOPQRSTUVWXYZ0123456789"
                            "\"{}:,.-_()[]+*?!'\\/|@#$%&=")
        json_allowed_tokens = set()
        for t_id, t_str in self.vocab.id_to_token.items():
            clean_str = t_str.replace("Ġ", "").replace("Ċ", "")
            # Adiciona o token se for apenas composto por caracteres permitidos
            if not clean_str or all(c in allowed_chars for c in clean_str):
                json_allowed_tokens.add(t_id)
        json_allowed_tokens.add(151645)
        strict_keys_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFG"
                                "HIJKLMNOPQRSTUVWXYZ0123456789_ \":,")
        strict_keys_tokens = set()
        for t_id, t_str in self.vocab.id_to_token.items():
            clean_str = t_str.replace("Ġ", "").replace("Ċ", "")
            if not clean_str or all(c in strict_keys_chars for c in clean_str):
                strict_keys_tokens.add(t_id)

        sample_logits = self.model.get_logits_from_input_ids([151645])
        # Compatibilidade caso venha em Tensor
        if hasattr(sample_logits, "tolist"):
            sample_logits = sample_logits.tolist()
        real_vocab_size = len(sample_logits)
        self.mask_json = np.full(real_vocab_size, -np.inf)
        for t_id in json_allowed_tokens:
            self.mask_json[t_id] = 0.0
        self.mask_strict_keys = np.full(real_vocab_size, -np.inf)
        for t_id in strict_keys_tokens:
            self.mask_strict_keys[t_id] = 0.0
        print("LLM Engine and Vocab ready!")

    def get_json_symbols_ids(self) -> dict[str, set[int]]:
        symbols = ['{', '}', ':', ',', '"']
        return {sym: self.vocab.find_tokens_for_char(sym) for sym in symbols}

    def custom_decode(self, token_ids: list[int]) -> str:
        # 1. Obter as strings cruas do dicionário
        raw_text = "".join(self.vocab.id_to_token.get(t_id, "")
                           for t_id in token_ids)
        # 2. Substituir os caracteres BPE pelos normais
        clean_text = raw_text.replace("Ġ", " ").replace("Ċ", "\n")
        return clean_text

    def generate(self, prompt: str, max_tokens: int = 75) -> str:
        print("\n--- Intercepting Model ---")
        # Transforms the prompt text in ID's the llm understands
        input_tensor = self.model.encode(prompt)
        input_ids = input_tensor[0].tolist()
        prefix = '{"name":"'
        prefix_tensor = self.model.encode(prefix)
        prefix_ids = prefix_tensor[0].tolist()
        input_ids.extend(prefix_ids)
        generated_ids: list[int] = list(prefix_ids)
        open_brackets = 1
        generated_text = prefix
        inside_string = True
        last_structural_char = '{'
        prev_char = ''
        print(prefix, end="", flush=True)
        for _ in range(max_tokens):
            stripped_text = generated_text.strip()
            # --- EXTREME INFERENCE SKIPPING (Fast-Forwarding) ---
            if (stripped_text.endswith('"name')
                    or stripped_text.endswith('"parameters')):
                next_token_id = self.t_quote
            elif (stripped_text.endswith('"name"')
                    or stripped_text.endswith('"parameters"')):
                next_token_id = self.t_colon
            elif stripped_text.endswith('"parameters":'):
                next_token_id = self.t_open_brace
            elif stripped_text.endswith('"name":'):
                next_token_id = self.t_quote
            elif not inside_string and stripped_text.endswith('{'):
                next_token_id = self.t_quote
            elif not inside_string and stripped_text.endswith(','):
                next_token_id = self.t_quote
            # --- FIM DO SKIPPING (Acordar a IA) ---
            else:
                # O modelo só "pensa" quando tem liberdade de escolha real
                raw_logits = (
                    self.model.get_logits_from_input_ids(input_ids)
                )
                if hasattr(raw_logits, "detach"):
                    logits = raw_logits.detach().cpu().numpy()
                elif hasattr(raw_logits, "numpy"):
                    logits = raw_logits.numpy()
                else:
                    logits = np.array(raw_logits.tolist()
                                      if hasattr(raw_logits, "tolist")
                                      else raw_logits)
                # --- MINI-FSM DE SEGURANÇA (Schema Enforcement) ---
                if (not last_structural_char
                        or last_structural_char in '{,'):
                    masked_logits = logits + self.mask_strict_keys
                else:
                    masked_logits = logits + self.mask_json
                next_token_id = int(np.argmax(masked_logits))
            if next_token_id == 151645:
                break
            generated_ids.append(next_token_id)
            input_ids.append(next_token_id)
            raw_word = self.vocab.id_to_token.get(next_token_id, "")
            clean_word = raw_word.replace("Ġ", " ").replace("Ċ", "\n")
            print(clean_word, end="", flush=True)
            generated_text += clean_word
            for char in clean_word:
                if char == '"':
                    escaped = (prev_char == '\\')
                    if not escaped:
                        inside_string = not inside_string
                if not inside_string:
                    if char == '{':
                        open_brackets += 1
                        last_structural_char = char
                    elif char == '}':
                        open_brackets -= 1
                        last_structural_char = char
                    elif char in ':,':
                        last_structural_char = char
                prev_char = char
            if open_brackets == 0:
                break
        print("\n\n---Generation Completed---")
        return str(self.custom_decode(generated_ids))
