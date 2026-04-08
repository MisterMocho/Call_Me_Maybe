from pydantic import BaseModel


class ParameterDefinition(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition


class TestPrompt(BaseModel):
    prompt: str
