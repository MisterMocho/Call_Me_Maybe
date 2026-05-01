"""
Constrained-decoding inference engine for the function-calling agent.

The engine wraps the raw small LLM provided by the SDK and intercepts
its generation loop token by token. Three cooperating mechanisms ensure
that the model only ever produces syntactically valid JSON conforming
to the function-calling format:

    1. Logit masking:    illegal tokens have their probability driven
                         to negative infinity before ``argmax``.
    2. Fast-forwarding:  when the next token is structurally
                         determined, the model is bypassed entirely.
    3. Brace counting:   generation halts the moment the JSON object
                         is closed, preventing trailing prose.

A small finite-state machine, encoded in three persistent pieces of
state (``inside_string``, ``last_structural_char``, ``prev_char``),
keeps these mechanisms aware of the current position within the JSON
grammar even when the BPE tokenizer splits or fuses characters across
token boundaries.
"""


import numpy as np
from pathlib import Path
from typing import Any
from llm_sdk import Small_LLM_Model
from src.vocab_loader import VocabLoader


class LLMEngine:
    """
    Constrained-decoding wrapper around the project's small LLM.

    On construction the engine loads the underlying model, reads its
    vocabulary, and pre-computes two logit masks used to enforce the
    function-calling JSON grammar at every step of generation:

        * ``mask_json``         — the broad mask applied inside string
                                  values; permits everything that may
                                  legitimately appear in a JSON value.
        * ``mask_strict_keys``  — the narrow mask applied immediately
                                  after ``{`` or ``,``, where only a
                                  JSON key may legally start.

    Both masks are dense ``numpy`` arrays the size of the model's
    vocabulary, with ``0.0`` for allowed tokens and ``-inf`` for
    forbidden ones. Masking is therefore a single vector addition at
    inference time.

    Attributes:
        model: The underlying SDK model. Typed as ``Any`` because the
            SDK does not expose a public type.
        vocab: BPE vocabulary lookup helper.
        t_quote, t_colon, t_open_brace: Token IDs reused by the
            fast-forwarding logic.
        mask_json: Dense logit mask used inside string values.
        mask_strict_keys: Dense logit mask used after ``{`` or ``,``.
    """
    def __init__(self) -> None:
        """
        Initialise the SDK model, load the vocabulary, and precompute
        the two logit masks used during generation.

        The constructor performs the one-time work that does not
        depend on the prompt: identifying structural token IDs,
        scanning the vocabulary for tokens whose surface form lies
        within the legal JSON character set, and probing the model
        for its true vocabulary size so that mask arrays are
        correctly sized.
        """

        print("Initializing LLM engine...")
        self.model: Any = Small_LLM_Model()
        vocab_path_str: str = str(self.model.get_path_to_vocab_file())
        self.vocab = VocabLoader(Path(vocab_path_str))
        self.t_quote = self._require_token_id('"')
        self.t_colon = self._require_token_id(':')
        self.t_open_brace = self._require_token_id('{')
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEF"
                            "GHIJKLMNOPQRSTUVWXYZ0123456789"
                            "\"{}:,.-_()[]+*?!'\\/|@#$%&=")
        json_allowed_tokens = set()
        for t_id, t_str in self.vocab.id_to_token.items():
            clean_str = t_str.replace("Ġ", "").replace("Ċ", "")
            # Adds the token only if made by allowed characters
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
        # Compatibility in case it is a tensor
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

    def _require_token_id(self, token: str) -> int:
        """Look up a token ID and fail loudly if it is missing."""
        tid = self.vocab.get_token_id(token)
        if tid is None:
            raise ValueError(f"Required token not found: {token!r}")
        return tid

    def custom_decode(self, token_ids: list[int]) -> str:
        """
        Convert a list of token IDs back into a human-readable string.

        The vocabulary is consulted directly rather than using the
        SDK's decoder, which avoids pulling in additional model
        machinery. BPE markers are normalised to their textual
        equivalents: ``Ġ`` becomes a space, ``Ċ`` becomes a newline.

        Args:
            token_ids: List of integer IDs produced by the engine.

        Returns:
            The decoded string with BPE markers replaced.
        """

        # 1. Gets the raw strings from the dictionary
        raw_text = "".join(self.vocab.id_to_token.get(t_id, "")
                           for t_id in token_ids)
        # 2. Replaces all BPE characters by normal ones
        clean_text = raw_text.replace("Ġ", " ").replace("Ċ", "\n")
        return clean_text

    def generate(self, prompt: str, max_tokens: int = 75) -> str:
        """
        Run constrained generation and return the decoded JSON string.

        The method wraps the prompt with the deterministic JSON prefix
        ``{"name":"`` and then enters a token-by-token loop. At each
        step one of two things happens:

            * If the next character is structurally determined by the
              JSON grammar (the ``:`` after a key, the ``{`` after
              ``"parameters":``, etc.) the corresponding token ID is
              injected directly without consulting the model.
            * Otherwise the model is asked for its logits, which are
              then masked according to whether the cursor is at a
              position that may start a key (after ``{`` or ``,``)
              or anywhere else, and the highest-scoring legal token
              is chosen.

        Three pieces of state persist for the whole generation and
        across token boundaries:

            * ``inside_string``       — toggles on every unescaped
                                        ``"``.
            * ``last_structural_char``— the most recent structural
                                        JSON character (``{``, ``}``,
                                        ``:``, ``,``) seen outside a
                                        string.
            * ``prev_char``           — the character immediately
                                        emitted, used to recognise
                                        escape sequences such as
                                        ``\\"`` even when the BPE
                                        tokenizer splits ``\\`` and
                                        ``"`` into separate tokens.

        Generation stops as soon as the brace counter returns to zero
        or the EOS token is produced, guaranteeing that no
        conversational prose can leak after the JSON closes.

        Args:
            prompt: The fully assembled system prompt, including the
                tool list and the current user query.
            max_tokens: Hard cap on the number of generation
                iterations. Acts as a safety net; the brace counter
                is the primary stopping condition.

        Returns:
            The decoded JSON string emitted by the engine. Always a
            valid JSON object by construction.
        """

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
            if (stripped_text.endswith('"parameters')):
                next_token_id = self.t_quote
            elif (stripped_text.endswith('"parameters"')):
                next_token_id = self.t_colon
            elif stripped_text.endswith('"parameters":'):
                next_token_id = self.t_open_brace
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
                if prev_char == '\\' and char == '\\':
                    prev_char = ''
                else:
                    prev_char = char
            if open_brackets == 0:
                break
        print("\n\n---Generation Completed---")
        return str(self.custom_decode(generated_ids))
