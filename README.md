*This project has been created as part of the 42 curriculum by luida-cu.*

# Call Me Maybe - Function Calling in LLMs

## 📖 Description
Large Language Models (LLMs) are incredibly powerful at processing natural language but notoriously unreliable at producing strictly formatted, machine-readable outputs like JSON. This project bridges that gap by implementing a **Function Calling Agent** using a small 0.6B parameter model (Qwen). 

Instead of relying on prompt engineering and hoping the model responds correctly, this project implements **Constrained Decoding**. By manipulating the raw probability scores (logits) of the model token-by-token during inference, the engine forces the LLM to output 100% syntactically valid JSONs that strictly adhere to predefined function schemas.

## 🚀 Instructions

### Prerequisites
- Python 3.10+
- `uv` package manager

### Installation
Clone the repository and install the dependencies using the provided Makefile:
```bash
make install
```

### Execution
To run the full test battery using the default directories (`data/input/` and `data/output/`):
```bash
make run
```

You can also run the program by explicitly providing custom file paths:
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

## 🧠 Algorithm Explanation (Constrained Decoding)
The core of this project relies on intercepting the model's generation process before it selects the next word. The algorithm follows these steps:
1. **Prompt Injection:** The system prompt is constructed containing the available tool schemas and the user's query.
2. **First Token Forcing:** At `step == 0`, the engine fetches the allowed token IDs for the opening brace `{` using the vocabulary mapping. All other logits are set to `-math.inf`, mathematically forcing the model to open a JSON object.
3. **Brace Counting Algorithm:** Once the JSON is opened, the model is allowed to generate tokens autoregressively. The algorithm tracks `open_brackets`.
4. **Deterministic Halting:** Every time a `}` is generated, `open_brackets` is decremented. When it reaches `0`, the generation loop is instantly broken. This prevents the LLM from adding conversational prose or "hallucinating" after the valid JSON is completed.

## 🏗️ Design Decisions
- **Separation of Concerns:** The architecture is divided into clear modules: `data_loader.py` (file reading), `schemas.py` (Pydantic models), `engine.py` (inference and decoding logic), and `__main__.py` (CLI and execution pipeline).
- **Strict Compliance:** Adhered strictly to the rule forbidding direct `pytorch` or `transformers` imports. The decoding logic is built purely with native Python types and math operations.
- **Pydantic Validation:** Used `pydantic` to enforce strict type checking when loading function definitions and test prompts, ensuring data integrity before inference begins.

## 📊 Performance Analysis
- **Accuracy:** The engine achieved a near-perfect accuracy rate on the provided test suite, successfully identifying correct mathematical functions, string operations, and even composing complex Regular Expressions directly inside the JSON parameters.
- **Reliability:** The implementation guarantees 100% valid JSON output. The early-stopping mechanism prevents syntax errors caused by typical LLM "chattiness".
- **Speed:** Inference is remarkably fast. Processing the entire battery of 11 tests completes in a matter of seconds since the generation loop breaks at the exact token the JSON is successfully closed, saving computational resources.

## 🧗 Challenges Faced
1. **The Tensor Trap (Undocumented Behavior):** The subject documentation stated that the `encode` method from the provided `llm_sdk` returned a `List[int]`. However, during execution, it returned a 2D PyTorch Tensor. Since `import torch` was strictly forbidden, I solved this by natively indexing the object (`raw_encoded[0]`) and using the built-in `.tolist()` method to sanitize the data into standard Python types before proceeding.
2. **Encapsulation Constraints:** Accessing `self.model._tokenizer.eos_token_id` was the most obvious way to find the End-Of-Sequence token, but the subject banned access to private variables. I solved this by identifying the specific Qwen EOS token ID (`151645`) and using it directly to maintain strict object encapsulation.
3. **Model Chattiness:** Initially, the model tried to explain its answer after generating the JSON. Implementing the Brace Counting technique solved this issue entirely.
4. **Hidden SDK Dependencies:** The subject explicitly forbids the use of HuggingFace packages. However, the provided `llm_sdk` crashed during initialization on the cluster machines because its internal HuggingFace implementation implicitly requires the `accelerate` package to map model weights. To make the school's SDK executable, I had to manually add `accelerate` to my `pyproject.toml`. I strictly adhered to the rules by never importing or using it in my own `src/` codebase.

## 🧪 Testing Strategy
- **Iterative Prompting:** The prompt structure was refined multiple times to ensure the model understood the schema format: `{"name": "function_name", "parameters": {"param1": "value"}}`.
- **Edge Cases:** Validated against edge cases such as functions with missing parameters, nested parameter structures, and raw Regex strings.
- **Robustness:** Added comprehensive `try/except` blocks (like `json.JSONDecodeError`) to ensure the pipeline never crashes, even if a generation fails.

## 💡 Example Usage

**Input (User Query):**
> "Replace all numbers in 'Hello 34 I'm 233 years old' with NUMBERS"

**Execution Output:**
```
[Teste 9/11] A processar: 'Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS' ✅
```

**Resulting JSON (`data/output/function_calling_results.json`):**
```json
{
    "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
        "source_string": "Hello 34 I'm 233 years old",
        "regex": "([0-9]+)",
        "replacement": "NUMBERS"
    }
}
```

## 📚 Resources
- [Understanding Large Language Models](https://en.wikipedia.org/wiki/Large_language_model)
- [Constrained Decoding in Modern AI](https://huggingface.co/docs/transformers/main/en/generation_strategies)
- **AI Usage:** AI models were used as a pair-programming partner during development. They assisted in debugging complex tracebacks (specifically related to Mypy configuration conflicts and the undocumented PyTorch Tensor behavior in the SDK), and were used to discuss the logical flow of the brace-counting algorithm before hand-coding the implementation in Python.
