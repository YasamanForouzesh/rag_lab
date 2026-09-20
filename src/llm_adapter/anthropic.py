from .baseAdapter import BaseAdapter
import anthropic
from pydantic import BaseModel, ValidationError
from typing import Type
import json
from anthropic import transform_schema
from pydantic import TypeAdapter
import models as m

class Anthropic(BaseAdapter):
    def __init__(self, model: str, max_tokens: int = 4096):
        super().__init__(model, max_tokens)
        self.client = anthropic.Anthropic()

    def input_normalizer(
        self,
        items: list[m.Prompt | m.ToolCall | m.ToolResult],
        ) -> list[dict]:

        if not items:
            raise ValueError("At least one input item is required.")

        normalized = []
        tool_calls = []
        tool_results = []

        for index, item in enumerate(items):

            if isinstance(item, m.Prompt):

                if tool_calls:
                    raise ValueError(
                        "ToolCall must be followed by ToolResult "
                        "before another prompt."
                    )

                if tool_results:
                    normalized.append({
                        "role": "user",
                        "content": tool_results,
                    })
                    tool_results = []

                if item.role not in {"user", "assistant"}:
                    raise ValueError(
                        f"Invalid role '{item.role}' at index {index}"
                    )

                normalized.append({
                    "role": item.role,
                    "content": item.content,
                })

            elif isinstance(item, m.ToolCall):

                tool_calls.append({
                    "type": "tool_use",
                    "id": item.id,
                    "name": item.name,
                    "input": item.arguments,
                })

            elif isinstance(item, m.ToolResult):

                if tool_calls:
                    normalized.append({
                        "role": "assistant",
                        "content": tool_calls,
                    })
                    tool_calls = []

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": item.call_id,
                    "content": self.serialize_tool_result(item.result),
                })

            else:
                raise TypeError(
                    f"Unsupported input type: {type(item)}"
                )

        if tool_calls:
            normalized.append({
                "role": "assistant",
                "content": tool_calls,
            })

        if tool_results:
            normalized.append({
                "role": "user",
                "content": tool_results,
            })

        return normalized



    def generate_schema_config(self, raw_schema: type[BaseModel]):
        

        # Optionally run Anthropic's transformer to fit Claude API constraints
        clean_schema = TypeAdapter(raw_schema).json_schema()
        clean_schema = transform_schema(clean_schema)

        return {
            "format": {
                "type": "json_schema",
                "schema": clean_schema,
            }
        }




    def tool_normalizer(self, tools:list)-> list[dict]:
        normalized = []
        for tool in tools:
            if isinstance(tool,m.Tool):
                normalized.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters
            })
            elif isinstance(tool, dict):
                normalized.append(tool)

            else:
                raise TypeError(f"Unsupported tool type: {type(tool)}")

        return normalized


    
    def generate(
        self,
        prompt: list[m.Prompt | m.ToolCall | m.ToolResult],
        system_prompt: str | None = None,
        output_schema: Type[BaseModel] | None = None,
        tools: list[m.Tool | dict] | None = None
    ) -> m.LLMResponse:
        
        validated_messages = self.input_normalizer(prompt)


        request_args = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": validated_messages,
        }

        if system_prompt:
            request_args["system"] = system_prompt

        if tools:
            request_args["tools"] = self.tool_normalizer(tools)

        if output_schema:
          request_args["output_config"] = self.generate_schema_config(output_schema)

        # the create use the output_config which we have to manually format that
        # the parse use output_format that we can pass pydantic 
        # for streaming we should use create
        response = self.client.messages.create(**request_args)
        tool_calls = []

        for block in response.content:
            if block.type == "tool_use":

                tool_calls.append(
                    m.ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        raw_text = "".join(
            block.text
            for block in response.content
                if block.type == "text"
        )

        text = raw_text or None
        parsed = None

        if output_schema and raw_text:
            try:
                parsed = output_schema.model_validate_json(raw_text)
                text = None
            except (json.JSONDecodeError, ValidationError):
                text = raw_text

        return m.LLMResponse(
            text=text,
            parsed=parsed,
            tool_calls=tool_calls,
        )



