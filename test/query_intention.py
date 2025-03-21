from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI
import json
import os

# tagging_prompt = ChatPromptTemplate.from_template(
# """
# Extract the desired information from the following passage.

# Only extract the properties mentioned in the 'Classification' function.

# Passage:
# {input}
# """
# )


# class Classification(BaseModel):
#     keyword: List[str] = Field(description=u"查询的关键字列表")
#     rating: List[int] = Field(description="关键字的查询意图打分从1到10")

extraction_prompt = ChatPromptTemplate.from_template(
"""
分析以下问题，提取问题的关键字，如果有多个，按询问的意图高低排序。
<问题>
{input}
</问题>
"""
)

class Classification(BaseModel):
    keyword: List[str] = Field(description=u"查询的关键字")

dashscope_api_key = "sk-fe19314cfe7d424d9c8ec7c52f3dcd88"
agent_model = "qwen-max-latest"
base_url = os.getenv("OPENAI_BASEURL", default="https://dashscope.aliyuncs.com/compatible-mode/v1")

# LLM\n",
# llm = ChatTongyi(temperature=0.6, model=model, api_key=dashscope_api_key, streaming=True).with_structured_output(
#     Classification
# )

# # query = u"关于工业互联网历年有什么不同表述"
# query = u"历年来关于工业互联网有什么不同表述"
# # prompt = tagging_prompt.invoke({"input": query})
# # print(prompt)
# output = llm.invoke(query)
# print(output, type(output), output.keyword)

llm_with_tools = OpenAI(base_url=base_url, api_key=dashscope_api_key)

def extra_keyword(query) -> List[str]:
    prompt = extraction_prompt.invoke({"input": query})
    content = prompt.to_messages()[0].content
    print(content)

    schema = Classification.model_json_schema()
    func_schema = {
        "name": "Classification",
        "parameters": schema,
    }

    # print(json.dumps(func_schema, indent=2))
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

    # 解析工具调用结果
    try:
        # 确保正确访问工具调用结果
        tool_call = response.choices[0].message.tool_calls[0]
        arguments = tool_call.function.arguments

        try:
            arguments_dict = json.loads(arguments)
        except json.JSONDecodeError:
            print("尝试修复 JSON 格式问题...")

        # 如果 arguments_dict 为空，创建默认结构
        if not arguments_dict:
            arguments_dict = {"error": "模型返回了空的 JSON 结构"}

        print(tool_call)
        print(arguments_dict)
        return arguments_dict['keyword']

    except Exception as e:
        raise Exception("无法解析模型返回的工具调用: " + str(e))    

while True:
    user_query = input('请输入您的问题:（exit=退出）')
    if user_query == 'exit':
        print("再见!")
        break
    else:
        print(extra_keyword(user_query))
