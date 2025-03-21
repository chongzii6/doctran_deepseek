from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import os
from openai import OpenAI

agent_model = os.getenv("AGENT_MODEL", default="qwen-max-latest")
dashscope_api_key = os.getenv("DASHSCOPE_KEY")
base_url = os.getenv("OPENAI_BASEURL", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
agent_model = "qwen-max-2025-01-25"


class add(BaseModel):
    """Add two integers."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class multiply(BaseModel):
    """Multiply two integers."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")

# llm_with_tools = ChatOpenAI(temperature=0, model=agent_model, base_url=base_url, api_key=dashscope_api_key)
# llm_with_tools.bind_tools([add, multiply])
llm_with_tools = OpenAI(base_url=base_url, api_key=dashscope_api_key)
query = "What is 3 * 12?"
# query = "What is 3 * 12? Also, what is 11 + 49?"

# print(multiply.model_json_schema())
import json
schema = multiply.model_json_schema()
func_schema = {
    "name": "multiply",
    "parameters": schema,
}

print(json.dumps(func_schema, indent=2))

# response = llm_with_tools.invoke(query)

response = llm_with_tools.chat.completions.create(temperature=0, model=agent_model, 
            messages=[{"role": "user", "content": query}],
            tools=[{
                "type": "function",
                "function": func_schema,
                }
            ],
            tool_choice={"type": "function", "function": {"name": "multiply"}}
)
# response = llm_with_tools.chat.completions.create(temperature=0, model=agent_model, 
#             messages=[{"role": "user", "content": query}],
#             tools=[{
#                 "type": "function",
#                 "function": {
#                     "name": "multiply",
#                     "description": "Multiply two integers.",
#                     "parameters": {
#                            "type": "object",
#                            "properties": {
#                                "a": {
#                                    "type": "integer",
#                                    "description": "First integer",
#                                },
#                                "b": {
#                                    "type": "integer",
#                                    "description": "Second integer",
#                                },
#                            },
#                            "required": ["a", "b"],
#                         }
#                     },
#                 }
#             ],
#             tool_choice={"type": "function", "function": {"name": "multiply"}}
# )
print(response)
