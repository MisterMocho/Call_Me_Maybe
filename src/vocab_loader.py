"""
Load the BPE vocabulary file shipped with the model and expose lookups
in both directions.

The vocabulary is read once at startup from the path returned by the
SDK's ``get_path_to_vocab_file()``. Two dictionaries are kept in memory
so that the engine can resolve token strings to IDs (when building
masks of allowed tokens) and IDs back to strings (when decoding
generated output).

The leading ``Ġ`` byte commonly found in BPE vocabularies represents a
preceding space and is treated transparently by ``find_tokens_for_char``.
"""


import json
from pathlib import Path


class VocabLoader:
    """
    In-memory representation of a BPE vocabulary.

    The class keeps two mirrored dictionaries that allow both
    ``token -> id`` and ``id -> token`` lookups in O(1). It is
    intentionally minimal: no encoding or decoding logic lives here —
    those concerns belong to the engine, which has the surrounding
    context (special tokens, BPE markers) needed to interpret the
    raw strings.

    Attributes:
        vocab_path: Path to the JSON file the vocabulary was loaded
            from. Stored only for diagnostics.
        token_to_id: Mapping from raw vocabulary token (including BPE
            markers like ``Ġ``) to integer ID.
        id_to_token: Inverse of ``token_to_id``, populated at load
            time so that decoding never has to scan the table.
    """

    def __init__(self, vocab_path: Path) -> None:
        """
        Build the in-memory vocabulary from a JSON file.

        Args:
            vocab_path: Path to a JSON file mapping token strings to
                integer IDs, as produced by HuggingFace tokenizers.
                Loaded eagerly inside this constructor.
        """

        self.vocab_path = vocab_path
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._load_vocab()

    def _load_vocab(self) -> None:
        """
        Read the vocabulary file and populate the two lookup tables.

        The file is expected to be a JSON object whose keys are the
        token strings and whose values are the matching integer IDs.
        The reverse mapping is built immediately so that subsequent
        lookups are constant-time in either direction.
        """

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {value: k for k, value in self.token_to_id.items()}

    def get_token_id(self, token_str: str) -> int | None:
        """
        Resolve a raw vocabulary token to its integer ID.

        Args:
            token_str: The exact token string as it appears in the
                vocabulary, including any BPE marker prefix.

        Returns:
            The integer ID if the token is present, otherwise ``None``.
        """

        return self.token_to_id.get(token_str)

    def find_tokens_for_char(self, char: str) -> set[int]:
        """
        Return all token IDs whose surface form is a given single
        character, with or without a leading BPE space marker.

        This is the primary tool the engine uses to build the masks of
        structurally meaningful tokens such as ``{``, ``"`` and ``:``.
        Both the bare form (``"{"``) and the space-prefixed form
        (``"Ġ{"``) are matched, because either may legitimately follow
        the previous token depending on context.

        Args:
            char: A single character whose token IDs are wanted.

        Returns:
            A set of integer token IDs. May be empty if the character
            is not represented in the vocabulary.
        """

        valid_ids: set[int] = set()
        for token_str, token_id in self.token_to_id.items():
            if token_str == char or token_str == f"Ġ{char}":
                valid_ids.add(token_id)
        return valid_ids
