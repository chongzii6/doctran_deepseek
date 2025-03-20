from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
from pydantic import BaseModel, Field
from typing import List

tagging_prompt = ChatPromptTemplate.from_template(
"""
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)


# class Classification(BaseModel):
#     keyword: List[str] = Field(description=u"查询的关键字列表")
#     rating: List[int] = Field(description="关键字的查询意图打分从1到10")

class Classification(BaseModel):
    keyword: str = Field(description=u"查询的关键字")

dashscope_api_key = "sk-fe19314cfe7d424d9c8ec7c52f3dcd88"
model = "qwen-max-latest"

# LLM\n",
llm = ChatTongyi(temperature=0.6, model=model, api_key=dashscope_api_key, streaming=True).with_structured_output(
    Classification
)

# query = u"关于工业互联网历年有什么不同表述"
query = u"历年来关于工业互联网有什么不同表述"
# prompt = tagging_prompt.invoke({"input": query})
# print(prompt)
output = llm.invoke(query)
print(output, type(output), output.keyword)
