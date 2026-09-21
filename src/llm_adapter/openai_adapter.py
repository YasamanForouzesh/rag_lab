from .baseAdapter import BaseAdapter
from openai import OpenAI
from pydantic import BaseModel
from typing import Type
from . import models as m
import json

class OpenAIAdapter(BaseAdapter):
    def __init__(self, model: str, max_tokens: int = 4096):
        super().__init__(model, max_tokens)
        self.client = OpenAI()



    def tool_normalizer(self, tools:list)-> list[dict]:
        normalized = []
        for tool in tools:
            if isinstance(tool,m.Tool):
                normalized.append({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": tool.strict,
                })
            elif isinstance(tool, dict):
                normalized.append(tool)

            else:
                raise TypeError(f"Unsupported tool type: {type(tool)}")

        return normalized



    def input_normalizer(self,
        items: list[m.Prompt | m.ToolCall | m.ToolResult]
    ) -> list[dict]:

        if not items:
            raise ValueError("At least one input item is required.")

        normalized = []

        for index, item in enumerate(items):

            if isinstance(item, m.Prompt):
                if item.role not in {"user", "assistant", "developer"}:
                    raise ValueError(
                        f"Invalid role '{item.role}' at index {index}"
                    )

                normalized.append({
                    "role": item.role,
                    "content": item.content,
                })

            elif isinstance(item, m.ToolCall):
                normalized.append({
                    "type": "function_call",
                    "call_id": item.id,
                    "name": item.name,
                    "arguments": json.dumps(item.arguments),
                })

            elif isinstance(item, m.ToolResult):

                normalized.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": self.serialize_tool_result(item.result),
                })

            else:
                raise TypeError(
                    f"Unsupported input type: {type(item)}"
                )

        return normalized


    def generate(
        self,
        prompt: list[m.Prompt | m.ToolCall | m.ToolResult],
        system_prompt: str | None = None,
        output_schema: Type[BaseModel] | None = None,
        tools: list[m.Tool | dict] | None = None
    ) -> m.LLMResponse:

        input_items = self.input_normalizer(prompt)

        request_args = {
            "model": self.model,
            "input": input_items,
        }

        if system_prompt:
            request_args["instructions"] = system_prompt

        if tools:
            request_args["tools"] = self.tool_normalizer(tools)
        if output_schema:
            request_args["text_format"] = output_schema

            response = self.client.responses.parse(**request_args)

            text = None
            parsed = response.output_parsed

        else:
            response = self.client.responses.create(**request_args)

            text = response.output_text
            parsed = None

        tool_calls = []
        print("response.output", response.output)
        for item in response.output:
            if item.type == "function_call":
                tool_calls.append(
                    m.ToolCall(
                        id=item.call_id,
                        name=item.name,
                        arguments=json.loads(item.arguments),
                    )
                )
            

        return m.LLMResponse(
            text=text,
            parsed=parsed,
            tool_calls=tool_calls,
        )




