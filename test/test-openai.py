from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI

import os

agent_model = os.getenv("AGENT_MODEL", default="qwen-max-latest")
dashscope_api_key = os.getenv("DASHSCOPE_KEY")
base_url = os.getenv("OPENAI_BASEURL", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
agent_model = "qwen-max-2025-01-25"

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function. 

passage:
{input}
"""
)

# output with JSON format.


# tagging_prompt = ChatPromptTemplate.from_template(
#     """
# Extract the desired information from the following passage.

# output with JSON format.

# Passage:
# {input}
# """
# )

json_schema = {
    "name": "Classification",
    "type": "object",
    "parameters": {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "description": "The sentiment of the text",
            },
            "aggressiveness": {
                "type": "integer",
                "description": "How aggressive the text is on a scale from 1 to 10",
            },
            "language": {
                "type": "string",
                "description": "The language the text is written in",
            },
        },
        "required": ["sentiment", "aggressiveness", "language"],
    },
    "properties": {
        "sentiment": {
            "type": "string",
            "description": "The sentiment of the text",
        },
        "aggressiveness": {
            "type": "integer",
            "description": "How aggressive the text is on a scale from 1 to 10",
        },
        "language": {
            "type": "string",
            "description": "The language the text is written in",
        },
    },
    "required": ["sentiment", "aggressiveness", "language"],
}

from langchain_core.tools import tool

class Classification(BaseModel):
    sentiment: str = Field(..., description="The sentiment of the text")
    aggressiveness: int = Field(..., 
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(..., description="The language the text is written in")

import json
schema = Classification.model_json_schema()
func_schema = {
    "name": "Classification",
    "parameters": schema,
}
# schema["name"] = schema.pop("title")

print(json.dumps(func_schema, indent=2))

# @tool
# def Classification(sentiment: str, aggressiveness: int, language: str):
#     return
#     # """Multiply a and b."""
#     # return a * b


# LLM

# llm = ChatOpenAI(temperature=0, model=agent_model, base_url=base_url, api_key=dashscope_api_key).with_structured_output(
#     # schema=Classification,
#     schema=json_schema,
#     method="json_mode",
#     # strict=False,
# )
# # llm = ChatOpenAI(temperature=0.6, model=agent_model, base_url=base_url, api_key=dashscope_api_key)
# # llm.bind_tools([Classification])

inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
prompt = tagging_prompt.invoke({"input": inp})
# print(prompt)
# response = llm.invoke(prompt)
content = prompt.to_messages()[0].content
print(content)

llm_with_tools = OpenAI(base_url=base_url, api_key=dashscope_api_key)

response = llm_with_tools.chat.completions.create(temperature=0, model=agent_model, 
            messages=[{"role": "user", "content": content}],
            tools=[{
                "type": "function",
                "function": func_schema,
                }
            ],
            # tool_choice={"type": "function", "function": {"name": schema["name"]}}
)

print(response)
