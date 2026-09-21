from dataclasses import dataclass
from typing import Any, Callable
from pydantic import BaseModel
from typing import Literal
from pydantic import Field


@dataclass
class WebSearchConfig:
    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    user_location: dict | None = None


@dataclass
class WebFetchConfig:
    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    citations: bool = True
    max_content_tokens: int | None = None



@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable
    strict: bool = False


    def __call__(self, *args, **kwds):
        return self.func(*args, **kwds)

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": self.strict,
            "additionalProperties": self.additionalProperties
        }


class Prompt(BaseModel):
    role: Literal["user", "assistant", "developer"]
    content: str


class ToolResult(BaseModel):
    call_id: str
    name: str
    result: Any

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict


class LLMResponse(BaseModel):
    text: str | None = None
    parsed: BaseModel | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)