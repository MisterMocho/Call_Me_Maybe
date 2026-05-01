"""
Pydantic models that describe the input contracts of the project.

Two JSON files are validated against these schemas at startup:
    * functions_definition.json -> list[FunctionDefinition]
    * function_calling_tests.json -> list[TestPrompt]

Validation happens once, in `data_loader.load_data`, before any inference
begins. This guarantees that downstream code (engine, prompt builder, type
caster) can rely on well-formed data without repeating defensive checks.
"""


from pydantic import BaseModel


class ParameterDefinition(BaseModel):
    """
    Describes the type contract of a single function parameter.

    Attributes:
        type: JSON-schema-style type identifier as declared in the
            functions definition file. Expected values include
            "string", "number", "integer", and "boolean". The value
            is consulted after generation to cast the raw JSON output
            to the type expected by the caller.
    """

    type: str


class FunctionDefinition(BaseModel):
    """
    Schema for a single tool/function the LLM is allowed to call.

    Attributes:
        name: Canonical function identifier emitted by the model
            (e.g. ``fn_add_numbers``). Used both to drive prompt
            construction and to look the function up after generation.
        description: Human-readable summary of what the function does.
            Currently omitted from the prompt for performance, but kept
            in the schema for extensibility.
        parameters: Mapping from parameter name to its
            :class:`ParameterDefinition`. Order is preserved as supplied
            by the loader.
        returns: Type contract of the function's return value. Not used
            by the engine but validated for completeness of the schema.
    """

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition


class TestPrompt(BaseModel):
    """
    A single user prompt to be evaluated by the engine.

    Attributes:
        prompt: Natural-language instruction issued by the user.
            Passed verbatim into the system prompt; the engine is
            expected to map it onto exactly one tool from the
            functions definition.
    """

    prompt: str
